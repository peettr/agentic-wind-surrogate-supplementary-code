"""MultiScaleConv — Multi-scale feature aggregation network.

Processes input at multiple scales in parallel using different dilation rates,
then fuses features with channel attention. Inspired by ASPP (Atrous Spatial
Pyramid Pooling) but in a full U-Net structure.
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


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias=False), nn.GELU(),
            nn.Conv2d(ch // reduction, ch, 1, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class MultiScaleBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2, bias=False)
        self.conv4 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4, bias=False)
        self.conv8 = nn.Conv2d(ch, ch, 3, padding=8, dilation=8, bias=False)
        self.proj = nn.Conv2d(ch * 4, ch, 1, bias=False)
        self.gn1 = _gn(ch)
        self.gn2 = _gn(ch)
        self.gn4 = _gn(ch)
        self.gn8 = _gn(ch)
        self.ca = ChannelAttention(ch)
        self.norm = _gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.gelu(self.gn1(self.conv1(x)))
        x2 = F.gelu(self.gn2(self.conv2(x)))
        x4 = F.gelu(self.gn4(self.conv4(x)))
        x8 = F.gelu(self.gn8(self.conv8(x)))
        out = self.proj(torch.cat([x1, x2, x4, x8], dim=1))
        return x + self.ca(self.norm(out))


class MultiScaleConv(BaseSurrogate):
    """Multi-scale feature aggregation for wind field prediction.

    Args:
        n_c: base channel count.
        depth: U-Net depth.
    """

    def __init__(self, n_c: int = 48, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, n_c, 4, stride=2, padding=1, bias=False), _gn(n_c), nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(MultiScaleBlock(ch), MultiScaleBlock(ch)))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            ch *= 2

        self.bottleneck = nn.Sequential(MultiScaleBlock(ch), MultiScaleBlock(ch))

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False), _gn(ch // 2), nn.GELU()))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False), _gn(ch // 2), nn.GELU(),
                MultiScaleBlock(ch // 2),
            ))
            ch //= 2

        self.output_proj = nn.Sequential(
            nn.Conv2d(n_c, n_c, 3, padding=1, bias=False), _gn(n_c), nn.GELU(),
            nn.Conv2d(n_c, 1, 1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]  # save original resolution
        x = self.input_proj(x)
        skips = []
        for blocks, down in zip(self.enc_blocks, self.down):
            x = blocks(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh != 0 or dw != 0:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[k](x)
        return F.interpolate(self.output_proj(x), size=(H, W), mode="bilinear", align_corners=False)
