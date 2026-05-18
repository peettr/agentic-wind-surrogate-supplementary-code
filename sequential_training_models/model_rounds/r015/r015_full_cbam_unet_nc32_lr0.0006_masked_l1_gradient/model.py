"""SE/CBAM Attention UNet — channel + spatial attention on skip connections.

Combines Squeeze-and-Excitation (SE) channel attention with optional spatial
attention from CBAM. Lightweight addition to standard UNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSurrogate


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, ch: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(ch // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class CBAMBlock(nn.Module):
    """CBAM: channel attention (SE) + spatial attention."""

    def __init__(self, ch: int, reduction: int = 4) -> None:
        super().__init__()
        # Channel attention
        self.se = SEBlock(ch, reduction)
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(ch, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.se(x)
        sa = self.spatial(x)
        return x * sa


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    dh = skip.size(2) - x.size(2)
    dw = skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
    return torch.cat([x, skip], dim=1)


class CBAMUNet(BaseSurrogate):
    """UNet with CBAM (channel + spatial) attention on skip connections.

    Args:
        depth: number of encoder stages (5, 6, or 7).
        n_c: base channel count.
        use_spatial: whether to include spatial attention (True=CBAM, False=SE only).
    """

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, depth: int = 7, n_c: int = 32, use_spatial: bool = True) -> None:
        super().__init__()
        if depth not in self.SUPPORTED_DEPTHS:
            raise ValueError(f"depth must be in {self.SUPPORTED_DEPTHS}, got {depth}")
        self.depth = depth
        self.n_c = n_c

        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch_in = 1
        for k in range(depth):
            ch_out = n_c * 2 ** k
            self.enc.append(ConvBlock(ch_in, ch_out))
            self.pool.append(nn.MaxPool2d(2))
            ch_in = ch_out

        bottleneck_ch = n_c * 2 ** depth
        self.bottleneck = ConvBlock(ch_in, bottleneck_ch)

        # Attention modules for skip connections
        Attn = CBAMBlock if use_spatial else SEBlock
        self.skip_attn = nn.ModuleList()
        for k in range(depth):
            ch_skip = n_c * 2 ** k
            self.skip_attn.append(Attn(ch_skip))

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch_in = bottleneck_ch
        for k in reversed(range(depth)):
            ch_skip = n_c * 2 ** k
            self.up.append(nn.ConvTranspose2d(ch_in, ch_skip, 2, stride=2))
            self.dec.append(ConvBlock(ch_skip * 2, ch_skip))
            ch_in = ch_skip

        self.head = nn.Sequential(nn.Conv2d(n_c, 1, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for k, (enc_block, pool) in enumerate(zip(self.enc, self.pool)):
            x = enc_block(x)
            # Apply attention to skip
            x = self.skip_attn[k](x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            x = _pad_cat(x, skip)
            x = self.dec[k](x)
        return self.head(x)
