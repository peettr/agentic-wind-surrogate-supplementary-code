"""CNN-DeepONet — CNN branch encoder + MLP trunk for dense field regression.

Branch network (CNN) encodes the building height map into a latent vector.
Trunk network (MLP) queries arbitrary spatial locations to predict wind speed.
For dense prediction, trunk outputs a coarse grid (64x64) then bilinearly
upsamples to 640x640.

Based on: DeepONet (Lu et al., 2021); CNN-DeepONet variants.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSurrogate


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class CNNEncoder(nn.Module):
    """CNN branch: encodes input image to latent vector."""

    def __init__(self, in_ch: int, latent_dim: int, n_c: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, n_c, 4, stride=2, padding=1, bias=False),
            _gn(n_c), nn.ReLU(inplace=True),
            nn.Conv2d(n_c, n_c * 2, 4, stride=2, padding=1, bias=False),
            _gn(n_c * 2), nn.ReLU(inplace=True),
            nn.Conv2d(n_c * 2, n_c * 4, 4, stride=2, padding=1, bias=False),
            _gn(n_c * 4), nn.ReLU(inplace=True),
            nn.Conv2d(n_c * 4, n_c * 8, 4, stride=2, padding=1, bias=False),
            _gn(n_c * 8), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(n_c * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class MLPTrunk(nn.Module):
    """Trunk network: takes (x, y) coordinates and branch latent, outputs wind speed.

    Outputs a coarse grid of predictions.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, depth: int = 4,
                 out_grid: int = 64) -> None:
        super().__init__()
        self.out_grid = out_grid
        # Register coordinate grid
        coords = torch.linspace(-1, 1, out_grid)
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')
        self.register_buffer('grid_x', gx.reshape(1, out_grid, out_grid))
        self.register_buffer('grid_y', gy.reshape(1, out_grid, out_grid))

        # MLP: input = (x, y, latent) -> hidden -> ... -> 1
        layers = []
        in_dim = 2 + latent_dim  # x, y + branch latent
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B = latent.size(0)
        N = self.out_grid ** 2
        # Expand coordinates
        gx = self.grid_x.expand(B, -1, -1).reshape(B, N, 1)
        gy = self.grid_y.expand(B, -1, -1).reshape(B, N, 1)
        coords = torch.cat([gx, gy], dim=-1)  # (B, N, 2)

        # Expand latent to each spatial position
        latent_exp = latent.unsqueeze(1).expand(-1, N, -1)  # (B, N, latent_dim)

        # Concatenate and predict
        inp = torch.cat([coords, latent_exp], dim=-1)  # (B, N, 2+latent)
        out = self.mlp(inp)  # (B, N, 1)
        return out.reshape(B, 1, self.out_grid, self.out_grid)


class CNNDeepONet(BaseSurrogate):
    """CNN-DeepONet: CNN branch + MLP trunk for dense wind field prediction.

    Args:
        latent_dim: dimension of the branch latent vector.
        hidden_dim: trunk MLP hidden dimension.
        trunk_depth: number of trunk MLP layers.
        out_grid: coarse output grid size (upsampled to 640x640).
        n_c: CNN base channel count.
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512,
                 trunk_depth: int = 4, out_grid: int = 160, n_c: int = 32) -> None:
        super().__init__()
        self.branch = CNNEncoder(1, latent_dim, n_c)
        self.trunk = MLPTrunk(latent_dim, hidden_dim, trunk_depth, out_grid)
        self.out_grid = out_grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.branch(x)  # (B, latent_dim)
        coarse = self.trunk(latent)  # (B, 1, out_grid, out_grid)
        # Bilinear upsample to 640x640
        out = F.interpolate(coarse, size=(640, 640), mode='bilinear', align_corners=False)
        return F.relu(out)
