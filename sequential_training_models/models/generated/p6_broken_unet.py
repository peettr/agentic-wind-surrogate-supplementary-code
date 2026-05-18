import torch
import torch.nn as nn

class BrokenUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.enc = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, n_c, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_c, n_c, 3),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_c, n_c, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_c, out_channels, 3),
        )

    def forward(self, x):
        nan_mask = torch.isnan(x)
        x = torch.where(nan_mask, torch.zeros_like(x), x)
        x = self.enc(x)
        x = self.dec(x)
        return x
