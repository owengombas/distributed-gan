import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloaders.CelebaPartitioner import celeba_shape
from typing import Tuple
import os
import random
import numpy as np

celeba_z_dim = 100
ngf = 64


# https://github.com/AKASHKADEL/dcgan-celeba/blob/master/networks.py
class CelebaGenerator(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int] = celeba_shape, z_dim: int = celeba_z_dim) -> None:
        super(CelebaGenerator, self).__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(z_dim, ngf*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(ngf*8, ngf*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(ngf*4, ngf*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(ngf*2, ngf,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(ngf, image_shape[0],
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x))

        return x
