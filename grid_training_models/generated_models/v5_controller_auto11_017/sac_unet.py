"""Generated standalone Auto V5 model for sac_unet.

This generated file is the training source of truth for this run.
Runtime model construction is local to this file rather than registry delegation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from abc import ABC, abstractmethod


class BaseSurrogate(nn.Module, ABC):
    """Standalone BaseSurrogate copy for generated models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Auto V5 generated training source-of-truth models."""

    def check_shapes(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[1:] != (1, 640, 640):
            raise ValueError(f"Input shape mismatch: expected (B, 1, 640, 640), got {tuple(x.shape)}")
        if y.shape[1:] != (1, 640, 640):
            raise ValueError(f"Output shape mismatch: expected (B, 1, 640, 640), got {tuple(y.shape)}")



def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class SpatialAdaptiveConv(nn.Module):
    """Conv2d where kernel weights are predicted per-pixel from a context branch.

    Uses a practical grouped approach: standard conv + per-pixel affine modulation
    predicted from a context branch. This gives spatially-varying filter behavior
    without the prohibitive memory cost of full per-pixel kernels at 640x640.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, groups: int = 1) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.ks = kernel_size
        self.padding = kernel_size // 2
        # Base convolution
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2,
                              groups=groups, bias=False)
        # Context branch: predict per-pixel scale and shift
        self.ctx = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            _gn(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch * 2, 1, bias=False),  # scale + shift
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = self.conv(x)
        params = self.ctx(x)  # (B, out_ch*2, H, W)
        scale, shift = params.chunk(2, dim=1)  # each (B, out_ch, H, W)
        scale = 1.0 + torch.tanh(scale)  # scale centered around 1
        return out * scale + shift + self.bias.view(1, -1, 1, 1)


class SACConv(nn.Module):
    """Practical Spatial Adaptive Conv: standard conv + spatial attention modulation."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn = _gn(out_ch)
        # Spatial attention: predict per-pixel modulation from input
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        attn = self.spatial_attn(x)
        return self.gn(out * attn)


class SACConvBlock(nn.Module):
    """Two SACConv + ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SACConv(in_ch, out_ch),
            nn.ReLU(inplace=True),
            SACConv(out_ch, out_ch),
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


class SACUNet(BaseSurrogate):
    """UNet with Spatial Adaptive Convolution blocks.

    Args:
        depth: number of encoder stages (5, 6, or 7).
        n_c: base channel count.
    """

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, depth: int = 7, n_c: int = 12) -> None:
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
            self.enc.append(SACConvBlock(ch_in, ch_out))
            self.pool.append(nn.MaxPool2d(2))
            ch_in = ch_out

        bottleneck_ch = n_c * 2 ** depth
        self.bottleneck = SACConvBlock(ch_in, bottleneck_ch)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch_in = bottleneck_ch
        for k in reversed(range(depth)):
            ch_skip = n_c * 2 ** k
            self.up.append(nn.ConvTranspose2d(ch_in, ch_skip, 2, stride=2))
            self.dec.append(SACConvBlock(ch_skip * 2, ch_skip))
            ch_in = ch_skip

        self.head = nn.Sequential(nn.Conv2d(n_c, 1, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc_block, pool in zip(self.enc, self.pool):
            x = enc_block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            x = _pad_cat(x, skip)
            x = self.dec[k](x)
        return self.head(x)


class Model(SACUNet):
    """Training entrypoint for generated Auto V5 runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)
