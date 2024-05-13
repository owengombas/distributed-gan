from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloaders.CelebaPartitioner import celeba_shape
import os
import random
import numpy as np

ndf = 64

class CelebaDiscriminator(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int] = celeba_shape) -> None:
        super(CelebaDiscriminator, self).__init__()
        self.cv1 = nn.Conv2d(image_shape[0], ndf, kernel_size=4, stride=2, padding=1, bias=False) # (3, 64, 64) -> (64, 32, 32)
        self.cv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1 ) # (64, 32, 32) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(ndf*2) # spatial batch norm is applied on num of channels
        self.cv3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1) # (128, 16, 16) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        self.cv4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False) # (256, 8, 8) -> (512, 4, 4)
        self.bn4 = nn.BatchNorm2d(ndf* 8)
        self.cv5 = nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False) # (512, 4, 4) -> (1, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.cv1(x))
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, True)
        x = torch.sigmoid(self.cv5(x))
        return x.view(-1, 1).squeeze(1)
