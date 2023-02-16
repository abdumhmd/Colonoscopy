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

import numpy as np
from PIL import Image
import os
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import albumentations as A
import glob as glob


'''
Data set preparation checklist:
1. Images 
* Images from dataset have the same size, required for packing images to a batch.
* Images height and width are divisible by 32. This step is important for segmentation, because almost all models have skip-connections between encoder and decoder and all encoders have 5 downsampling stages (2 ^ 5 = 32). Very likely you will face with error when model will try to concatenate encoder and decoder features if height or width is not divisible by 32.
* Images have correct axes order. PyTorch works with CHW order, we read images in HWC [height, width, channels], don`t forget to transpose image.
2. Masks 
* Masks have the same sizes as images.
* Masks have only 0 - background and 1 - target class values (for binary segmentation).
* Even if mask don`t have channels, you need it. Convert each mask from HW to 1HW format for binary segmentation (expand the first dimension).
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class KvasirSEGDataset(Dataset):
    def __init__(self, root_dir, file_list, transform=None):
        self.root_dir = root_dir
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.file_list[idx])
        mask_path = os.path.join(self.root_dir, 'masks', self.file_list[idx])

        
        # Load image and mask as PIL Image and convert to RGB
        img = np.asarray(Image.open(img_path).resize((256,256)).convert('RGB'))
        mask = np.asarray(Image.open(mask_path).resize((256,256)).convert('L'))
        

        # Apply transformations to image and mask
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        # convert mask and image to tensors
        img = ToTensor()(img)
        mask = ToTensor()(mask)
    

        # convert mask and image tensors to float
        img = img.float()
        mask = mask.float()
        
        

        return img, mask

# # get all file names in Kvasir-SEG/images
# # file_list = os.listdir('/Users/abdu/Desktop/PhD Files/Spring Semester/Machine Learning/Project/semantic_segmentation/Kvasir-SEG/train/images')
# file_list = glob.glob('/Users/abdu/Desktop/PhD Files/Spring Semester/Machine Learning/Project/semantic_segmentation/Kvasir-SEG/train/images/*.jpg')

# # Keep the file name with out the path
# file_list = [os.path.basename(x) for x in file_list]

# # Define augmentation pipeline
# transform = A.Compose([
#     A.Resize(256, 256),
#     A.HorizontalFlip(p=0.5),
#     A.RandomRotate90(p=0.5),
#     ToTensorV2(),
# ])

# # Create dataset and dataloader
# dataset = KvasirSEGDataset('/Users/abdu/Desktop/PhD Files/Spring Semester/Machine Learning/Project/semantic_segmentation/Kvasir-SEG/train',file_list, transform=transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # get the first batch
# images, masks = next(iter(dataloader))

# # visualize the first image and mask from the batch
# import matplotlib.pyplot as plt
# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plt.imshow(images[0].permute(1, 2, 0))
# plt.subplot(1, 2, 2)
# plt.imshow(masks[0].squeeze(), cmap='gray')
# plt.show()


    