"""Generated standalone Grid model for dcn_unet.

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
        """Forward pass for Grid generated training source-of-truth models."""

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


class DeformableConvBlock(nn.Module):
    """Approximation of deformable conv using offset prediction + standard conv.

    Full DCNv3 requires custom CUDA kernels. This uses a pragmatic approximation:
    predict spatial offsets from input, apply them via grid_sample, then convolve.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Offset predictor: 2 channels per kernel position (3x3 = 9 * 2 = 18)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_ch, 18, 3, padding=1, bias=False),
            _gn(18),
            nn.ReLU(inplace=True),
        )
        # Modulation (mask per kernel position, projected to out_ch)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn = _gn(out_ch)

    def _deform_grid(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Apply deformable sampling via grid_sample."""
        B, _, H, W = x.shape
        # Create base grid
        gy, gx = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        grid = torch.stack([gx, gy], dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
        # Add offsets (take mean of 9 kernel positions for simplicity)
        offset_mean = offset.reshape(B, 2, 9, H, W).mean(dim=2)  # (B, 2, H, W)
        offset_mean = offset_mean.permute(0, 2, 3, 1)  # (B, H, W, 2)
        grid = grid + offset_mean
        # Normalize to [-1, 1] for grid_sample
        grid[..., 0] = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0
        return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = self.mask_conv(x)
        x_deformed = self._deform_grid(x, offset)
        out = self.conv(x_deformed)
        # Apply modulation
        out = out * self.mask_conv(x)
        return self.gn(out)


class DCNBlock(nn.Module):
    """Two deformable conv layers + ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DeformableConvBlock(in_ch, out_ch),
            nn.ReLU(inplace=True),
            DeformableConvBlock(out_ch, out_ch),
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


class DCNUNet(BaseSurrogate):
    """UNet with deformable convolutions (DCNv3-style approximation).

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
            self.enc.append(DCNBlock(ch_in, ch_out))
            self.pool.append(nn.MaxPool2d(2))
            ch_in = ch_out

        bottleneck_ch = n_c * 2 ** depth
        self.bottleneck = DCNBlock(ch_in, bottleneck_ch)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch_in = bottleneck_ch
        for k in reversed(range(depth)):
            ch_skip = n_c * 2 ** k
            self.up.append(nn.ConvTranspose2d(ch_in, ch_skip, 2, stride=2))
            self.dec.append(DCNBlock(ch_skip * 2, ch_skip))
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


class Model(DCNUNet):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



