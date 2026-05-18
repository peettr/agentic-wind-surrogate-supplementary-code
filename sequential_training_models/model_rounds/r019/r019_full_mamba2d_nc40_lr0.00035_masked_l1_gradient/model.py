"""2D-Mamba — True 2D state space model for spatial data.

Scans the 2D spatial domain in 4 directional sweeps (row-left-to-right,
row-right-to-left, column-top-to-bottom, column-bottom-to-top), merges
results. Avoids the directional bias of 1D flattening.

Based on: 2D SSM variants (VMamba, GroupMamba)
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


class SS2DDirection(nn.Module):
    """Simplified SSM scan in one direction."""

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.A = nn.Parameter(torch.randn(dim, d_state) * 0.1)
        self.B_proj = nn.Linear(dim, d_state, bias=False)
        self.C_proj = nn.Linear(d_state, dim, bias=False)
        self.D = nn.Parameter(torch.ones(dim) * 0.5)
        self.dt = nn.Parameter(torch.ones(dim) * 0.1)
        self.d_state = d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) — already flattened in scan direction
        B_batch, L, C = x.shape
        x_in = self.proj_in(x)

        # Simplified: use exponential moving average as SSM approximation
        dt = F.softplus(self.dt).unsqueeze(0).unsqueeze(0)  # (1, 1, C)
        A_decay = torch.sigmoid(self.dt.unsqueeze(0).unsqueeze(0))  # (1, 1, C)

        # Running average scan
        h = torch.zeros(B_batch, 1, C, device=x.device, dtype=x.dtype)
        outputs = []
        for i in range(L):
            h = A_decay * h + (1 - A_decay) * x_in[:, i:i+1, :]
            outputs.append(h)

        y = torch.cat(outputs, dim=1)  # (B, L, C)
        y = y + x_in * self.D.unsqueeze(0).unsqueeze(0)
        return self.proj_out(y)


class SS2DBlock(nn.Module):
    """2D SSM: scan in 4 directions and merge."""

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scan_lr = SS2DDirection(dim, d_state)  # left-to-right
        self.scan_rl = SS2DDirection(dim, d_state)  # right-to-left
        self.scan_tb = SS2DDirection(dim, d_state)  # top-to-bottom
        self.scan_bt = SS2DDirection(dim, d_state)  # bottom-to-top
        self.merge = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)

        # LR: row-major, left to right
        y_lr = self.scan_lr(x_norm)

        # RL: row-major, right to left
        y_rl = self.scan_rl(x_norm.flip(1))

        # TB: column-major, top to bottom
        x_tb = x.permute(0, 3, 2, 1).reshape(B, H * W, C)
        x_tb = self.norm(x_tb)
        y_tb = self.scan_tb(x_tb)

        # BT: column-major, bottom to top
        y_bt = self.scan_bt(x_tb.flip(1))

        # Merge: project back to (B, C, H, W)
        y = torch.cat([
            y_lr,
            y_rl.flip(1),
            y_tb,
            y_bt.flip(1),
        ], dim=-1)  # (B, H*W, C*4)
        y = self.merge(y)  # (B, H*W, C)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return y + residual


class Mamba2DBlock(nn.Module):
    """2D Mamba block: SS2D + FFN."""

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        self.ssm = SS2DBlock(dim, d_state)
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


class Mamba2D(BaseSurrogate):
    """2D-Mamba: UNet with 2D bidirectional SSM blocks.

    Args:
        depth: number of encoder stages.
        n_c: base channel count.
        d_state: SSM state dimension.
        n_ssm: number of Mamba2D blocks per stage.
    """

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, depth: int = 7, n_c: int = 32, d_state: int = 16,
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
            Mamba2DBlock(bottleneck_ch, d_state) for _ in range(n_ssm)
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
