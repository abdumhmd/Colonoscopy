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

from model.model import UNetLightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

model=UNetLightning(in_channels=3, out_channels=1, lr=1e-3, batch_size=8)

checkpoint_callback = ModelCheckpoint(
    monitor='valid_per_image_iou',
    dirpath='checkpoints/',
    filename='unet-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

early_stop_callback = EarlyStopping(
    monitor='valid_per_image_iou',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)

trainer = Trainer(accelerator='auto', max_epochs=10, callbacks=[checkpoint_callback, early_stop_callback])

trainer.fit(model)