"""CNN-DeepONet - CNN branch encoder + MLP trunk for dense field regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class ReflectionConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, bias: bool = False) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNEncoder(nn.Module):
    """CNN branch: encodes input image to latent vector."""

    def __init__(self, in_ch: int, latent_dim: int, n_c: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ReflectionConv2d(in_ch, n_c, 4, stride=2, bias=False),
            _gn(n_c),
            nn.ReLU(inplace=True),
            ReflectionConv2d(n_c, n_c * 2, 4, stride=2, bias=False),
            _gn(n_c * 2),
            nn.ReLU(inplace=True),
            ReflectionConv2d(n_c * 2, n_c * 4, 4, stride=2, bias=False),
            _gn(n_c * 4),
            nn.ReLU(inplace=True),
            ReflectionConv2d(n_c * 4, n_c * 8, 4, stride=2, bias=False),
            _gn(n_c * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(n_c * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class MLPTrunk(nn.Module):
    """DeepONet trunk: maps coords -> per-pixel basis dotted with branch latent.

    Standard DeepONet: trunk(coord) -> basis (latent_dim per pixel),
    output = <branch_latent, trunk_basis> per pixel. Memory scales with n_pts,
    not bsz * n_pts, which keeps 640x640 inputs runnable.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim * out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        bsz = latent.size(0)
        out_h, out_w = out_hw
        n_pts = out_h * out_w

        y = torch.linspace(-1, 1, out_h, device=latent.device, dtype=latent.dtype)
        x = torch.linspace(-1, 1, out_w, device=latent.device, dtype=latent.dtype)
        gy, gx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([gx, gy], dim=-1).reshape(n_pts, 2)

        basis = self.mlp(coords).reshape(n_pts, self.out_channels, self.latent_dim)
        out = torch.einsum("bk,nck->bcn", latent, basis)
        return out.reshape(bsz, self.out_channels, out_h, out_w)


class cnn_deeponet(nn.Module):
    """CNN-DeepONet: CNN branch + MLP trunk for dense wind field prediction."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, n_c: int = 16, depth: int = 7) -> None:
        super().__init__()
        latent_dim = 256
        hidden_dim = 256
        trunk_depth = max(1, min(depth, 4))

        self.branch = CNNEncoder(in_channels, latent_dim, n_c)
        self.trunk = MLPTrunk(latent_dim, hidden_dim, trunk_depth, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(x)
        x = torch.where(valid, x, torch.zeros_like(x))

        latent = self.branch(x)
        x = self.trunk(latent, x.shape[-2:])
        return F.relu(x)


