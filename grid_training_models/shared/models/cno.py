"""CNO-v2 â€” Convolutional Neural Operator v2 for 2D dense regression.

Learns mappings between function spaces using a sequence of strided convolutions
with residual connections. Unlike standard CNNs, CNO explicitly lifts to higher
dimensional spaces and applies composition of nonlinear integral operators.

Based on: Raonic et al., 2023 (Convolutional Neural Operator)
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


class CNOBlock(nn.Module):
    """CNO residual block: lift â†’ conv â†’ nonlinear â†’ project."""

    def __init__(self, ch: int, lift_mult: int = 2) -> None:
        super().__init__()
        lifted = ch * lift_mult
        self.block = nn.Sequential(
            nn.Conv2d(ch, lifted, 3, padding=1, bias=False),
            _gn(lifted), nn.GELU(),
            nn.Conv2d(lifted, lifted, 3, padding=1, bias=False),
            _gn(lifted), nn.GELU(),
            nn.Conv2d(lifted, ch, 3, padding=1, bias=False),
            _gn(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CNO(BaseSurrogate):
    """Convolutional Neural Operator for dense wind field prediction.

    U-shaped encoder-decoder with CNO residual blocks at each resolution.
    Features over-parameterized lifting within each block for better
    representation of nonlinear operators.

    Args:
        n_c: base channel count.
        depth: number of encoder/decoder stages.
        n_blocks: CNO blocks per stage.
        lift_mult: channel multiplier inside CNO block.
    """

    def __init__(self, n_c: int = 48, depth: int = 4, n_blocks: int = 2,
                 lift_mult: int = 2) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, n_c, 4, stride=2, padding=1, bias=False),
            _gn(n_c), nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(*[CNOBlock(ch, lift_mult) for _ in range(n_blocks)]))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            ch *= 2

        self.bottleneck = nn.Sequential(*[CNOBlock(ch, lift_mult) for _ in range(n_blocks)])

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False), _gn(ch // 2), nn.GELU()))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False), _gn(ch // 2), nn.GELU(),
                *[CNOBlock(ch // 2, lift_mult) for _ in range(n_blocks)],
            ))
            ch //= 2

        self.output_proj = nn.Sequential(
            nn.Conv2d(n_c, n_c, 3, padding=1, bias=False),
            _gn(n_c), nn.GELU(),
            nn.Conv2d(n_c, 1, 1),
            nn.ReLU(),
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



