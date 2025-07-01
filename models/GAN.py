import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):  # 28x28
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_dim),
            nn.Sigmoid() 
        )

    def forward(self, z):
        x = self.model(z)           # shape: (B, 784)
        return x.view(-1, 1, 28, 28)  # reshape to image

class Discriminator(nn.Module):
    def __init__(self, in_channels=784, hidden_dims = [1024, 512, 256, 1]):
        super().__init__()
        self.blocks = nn.ModuleList()
        # Fill this 
        in_channels = 784
        for h in hidden_dims:
            self.blocks.append(nn.Sequential(
                nn.Linear(in_channels, h),
                nn.LeakyReLU( negative_slope=0.2, inplace=True)
            ))
            in_channels = h
        #self.blocks.append(nn.Linear(in_channels, 1))

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten to (B, 784)
        # Fill this 
        for block in self.blocks:
            x = block(x) 
        #print('dis before sig',x.shape, x[-1])
        x = F.sigmoid(x)
        #print('dis after sig',x.shape, x[-1])
        return x.view(-1)  # shape: (B,)
