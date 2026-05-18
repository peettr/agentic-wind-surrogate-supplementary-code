import torch
import torch.nn as nn
import torch.nn.functional as F

class transolver(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        hidden = n_c

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, hidden, kernel_size=3),
            nn.GELU(),
        ]

        for _ in range(max(0, depth - 2)):
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(hidden, hidden, kernel_size=3),
                nn.GELU(),
            ]

        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden, out_channels, kernel_size=3),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        mask = torch.isfinite(x)
        x = torch.where(mask, x, torch.zeros_like(x))
        x = self.net(x)
        return x


