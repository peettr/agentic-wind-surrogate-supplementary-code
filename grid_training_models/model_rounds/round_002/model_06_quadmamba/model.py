"""Generated standalone Grid model for quadmamba.

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


class QuadScanBlock(nn.Module):
    """Quad-tree scan: process 4 quadrants independently then merge."""

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.D = nn.Parameter(torch.ones(dim) * 0.5)
        self.decay = nn.Parameter(torch.ones(dim) * 0.1)

    def _scan_region(self, x: torch.Tensor) -> torch.Tensor:
        """Simple EMA scan on a flattened region."""
        B, L, C = x.shape
        alpha = torch.sigmoid(self.decay).unsqueeze(0).unsqueeze(0)
        h = torch.zeros(B, 1, C, device=x.device, dtype=x.dtype)
        out = []
        for i in range(L):
            h = alpha * h + (1 - alpha) * x[:, i:i+1, :]
            out.append(h)
        return torch.cat(out, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # Ensure even dimensions
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h])
        _, _, Hp, Wp = x.shape

        hH, hW = Hp // 2, Wp // 2

        # Split into 4 quadrants
        q_tl = x[:, :, :hH, :hW].reshape(B, C, -1).permute(0, 2, 1)  # top-left
        q_tr = x[:, :, :hH, hW:].reshape(B, C, -1).permute(0, 2, 1)  # top-right
        q_bl = x[:, :, hH:, :hW].reshape(B, C, -1).permute(0, 2, 1)  # bottom-left
        q_br = x[:, :, hH:, hW:].reshape(B, C, -1).permute(0, 2, 1)  # bottom-right

        # Scan each quadrant
        q_tl = self._scan_region(self.proj_in(q_tl))
        q_tr = self._scan_region(self.proj_in(q_tr))
        q_bl = self._scan_region(self.proj_in(q_bl))
        q_br = self._scan_region(self.proj_in(q_br))

        # Merge: cross-quadrant mixing via concatenation + linear
        # Top half
        top = torch.cat([q_tl, q_tr], dim=1)  # (B, hH*hW*2, C)
        bot = torch.cat([q_bl, q_br], dim=1)
        full = torch.cat([top, bot], dim=1)  # (B, Hp*Wp, C)

        y = self.proj_out(full) + full * self.D.unsqueeze(0).unsqueeze(0)
        y = y.permute(0, 2, 1).reshape(B, C, Hp, Wp)
        if pad_h or pad_w:
            y = y[:, :, :H, :W]
        return y + residual


class QuadMambaBlock(nn.Module):
    """QuadMamba block: Quad scan + FFN."""

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        self.ssm = QuadScanBlock(dim, d_state)
        self.norm = _gn(dim)
        self.ff = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(x)
        x = x + self.ff(self.norm(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    dh = skip.size(2) - x.size(2)
    dw = skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
    return torch.cat([x, skip], dim=1)


class QuadMamba(BaseSurrogate):
    """QuadMamba: UNet with quad-tree selective scan SSM blocks.

    Args:
        depth: number of encoder stages.
        n_c: base channel count.
        d_state: SSM state dimension (unused in simplified version).
        n_ssm: number of QuadMamba blocks per stage.
    """

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, depth: int = 7, n_c: int = 20, d_state: int = 16,
                 n_ssm: int = 2) -> None:
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

        bottleneck_ch = ch_in  # = n_c * 2^(depth-1)
        self.bottleneck = nn.Sequential(*[
            QuadMambaBlock(bottleneck_ch, d_state) for _ in range(n_ssm)
        ])

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


class Model(QuadMamba):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



