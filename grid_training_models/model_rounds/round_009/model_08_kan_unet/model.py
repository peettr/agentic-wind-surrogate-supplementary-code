"""Generated standalone Grid model for kan_unet.

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
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class KANLayer(nn.Module):
    """Simplified KAN layer using piecewise linear B-spline approximation.

    Each input channel is transformed by a learnable univariate function
    approximated by a linear combination of basis functions (SiLU + residuals).
    """

    def __init__(self, in_ch: int, out_ch: int, grid_size: int = 5) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.grid_size = grid_size
        # Base linear transformation (SiLU-gated)
        self.base = nn.Linear(in_ch, out_ch)
        self.silu = nn.SiLU()
        # Spline weights: (out_ch, in_ch, grid_size + 1)
        self.spline_weight = nn.Parameter(torch.randn(out_ch, in_ch, grid_size) * 0.1)
        # Grid endpoints
        h = 2.0 / grid_size
        self.register_buffer('grid', torch.linspace(-1, 1 - h, grid_size))

    def _basis(self, x: torch.Tensor) -> torch.Tensor:
        """B-spline basis functions."""
        # x: (B, in_ch)
        # For each channel, compute distance to grid points
        x_expand = x.unsqueeze(-1)  # (B, in_ch, 1)
        grid = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_size)
        # B2 basis: max(0, 1 - |x - grid|)
        dists = 1.0 - (x_expand - grid).abs()  # (B, in_ch, grid_size)
        basis = F.relu(dists)
        return basis  # (B, in_ch, grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> reshape to (B*H*W, C)
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)

        # Base: SiLU-gated linear
        base_out = self.base(self.silu(x_flat))  # (N, out_ch)

        # Spline component
        basis = self._basis(x_flat)  # (N, in_ch, grid_size)
        # Einsum: (N, in_ch, grid_size) Ã— (out_ch, in_ch, grid_size) -> (N, out_ch)
        spline_out = torch.einsum('nig,oig->no', basis, self.spline_weight)

        out = base_out + spline_out  # (N, out_ch)
        return out.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, out_ch, H, W)


class KANConvBlock(nn.Module):
    """Conv block where the 1x1 channel mixing is done by a KAN layer."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # 3x3 depthwise conv for spatial features
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            _gn(in_ch),
        )
        # KAN-based channel mixing (replaces 1x1 conv)
        self.kan = KANLayer(in_ch, out_ch)
        self.gn = _gn(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.kan(x)
        return F.relu(self.gn(x))


class KANUNetBlock(nn.Module):
    """Two KANConvBlock layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            KANConvBlock(in_ch, out_ch),
            KANConvBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    dh = skip.size(2) - x.size(2)
    dw = skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
    return torch.cat([x, skip], dim=1)


class KANUNet(BaseSurrogate):
    """UNet with KAN (Kolmogorov-Arnold Network) channel mixing layers.

    Args:
        depth: number of encoder stages (5, 6, or 7).
        n_c: base channel count.
        grid_size: number of B-spline basis functions in KAN layers.
    """

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, depth: int = 7, n_c: int = 8, grid_size: int = 5) -> None:
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
            self.enc.append(KANUNetBlock(ch_in, ch_out))
            self.pool.append(nn.MaxPool2d(2))
            ch_in = ch_out

        bottleneck_ch = n_c * 2 ** depth
        self.bottleneck = KANUNetBlock(ch_in, bottleneck_ch)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch_in = bottleneck_ch
        for k in reversed(range(depth)):
            ch_skip = n_c * 2 ** k
            self.up.append(nn.ConvTranspose2d(ch_in, ch_skip, 2, stride=2))
            self.dec.append(KANUNetBlock(ch_skip * 2, ch_skip))
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


class Model(KANUNet):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



