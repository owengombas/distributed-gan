import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, 6*6*144) # Adjust the size accordingly if needed
        
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(144, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 96, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh() # CIFAR images are usually scaled to [-1, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 144, 6, 6) # Reshape to match the conv layers' expected input dimensions
        x = self.conv_layers(x)
        return x
