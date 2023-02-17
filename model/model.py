#
# Created on Thu Feb 16 2023
#
# The MIT License (MIT)
# Copyright (c) 2023 Abdurahman A. Mohammed
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import torch 
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import glob as glob
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import albumentations as A
from data.Colonoscopy import KvasirSEGDataset
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


# Unet model in pytorch lightning
class UNetLightning(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, lr=1e-4, batch_size=8):
        super(UNetLightning, self).__init__()
        self.save_hyperparameters()
        self.batch_size=batch_size
        self.lr = lr
        self.model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=in_channels,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=out_channels,                      # model output channels (number of classes in your dataset)
        )
        self.criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")     

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")


    def prepare_data(self):
        # get all file names
        all_files = glob.glob('Kvasir-SEG/images/*.jpg')
        root_dir = "Kvasir-SEG/"
        # keep the file names only
        all_files = [file.split('/')[-1] for file in all_files]

        # split the data into train, validation, and test
        train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(train_files, test_size=0.2, random_state=42)
        # Define augmentation pipeline
        train_transform = A.Compose([
                    A.Resize(256, 256),
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Normalize()
            ])

        val_transform = A.Compose([

                    A.Resize(256, 256),
            ])

        
        test_transform = A.Compose([

                    A.Resize(256, 256),

            ])
        # Create dataset    
        self.train_dataset = KvasirSEGDataset(root_dir,train_files, transform=train_transform)
        self.val_dataset = KvasirSEGDataset(root_dir,val_files, transform=val_transform)
        self.test_dataset = KvasirSEGDataset(root_dir,test_files, transform=test_transform)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, pin_memory=True, shuffle=False)

    def shared_step(self, batch, stage):
        
        image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0


        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4


        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.criterion(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
