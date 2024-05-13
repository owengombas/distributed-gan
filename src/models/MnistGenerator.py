import torch
import torch.nn as nn
from dataloaders.MnistPartitioner import mnist_shape
from typing import Tuple

mnist_z_dim = 100
ngf = 64


class MnistGenerator(nn.Module):
    def __init__(self, image_shape: Tuple[int, int, int] = mnist_shape, z_dim: int = mnist_z_dim):
        super(MnistGenerator, self).__init__()
        
        # Generator will up-sample the input producing input of size
        # suitable for feeding into discriminator
        self.fc1 = nn.Linear(in_features=z_dim, out_features=32)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)        
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)        
        self.fc4 = nn.Linear(in_features=128, out_features=mnist_shape[0] * mnist_shape[1] * mnist_shape[2])
        self.dropout = nn.Dropout(0.3)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the input
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # Feed forward
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)        
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu3(x)        
        x = self.dropout(x)
        x = self.fc4(x)
        tanh_out = self.tanh(x)

        # Reshape to image shape
        logit_out = tanh_out.view(-1, mnist_shape[0], mnist_shape[1], mnist_shape[2])
        
        return logit_out
