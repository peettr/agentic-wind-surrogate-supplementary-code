"""NAFNet — Nonlinear Activation Free Network for dense regression.

Uses SimpleGate (element-wise channel gating) and Simple Channel Attention (SCA)
instead of traditional nonlinear activations. Achieves SOTA on image restoration
benchmarks with clean, stable training.

Based on: NAFNet (Chen et al., 2022, ECCV)
Adapted: Simplified for single-input single-output dense regression, U-shaped architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSurrogate


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class SimpleGate(nn.Module):
    """Split channels in half, multiply: out = x[:C//2] * x[C//2:]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimpleChannelAttention(nn.Module):
    """Lightweight channel attention via global average pooling + 1x1 conv + sigmoid."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(ch, ch, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.conv(self.pool(x)))


class NAFBlock(nn.Module):
    """NAFNet residual block: SimpleGate + SCA, no nonlinear activation.

    Architecture (per block):
      x → 1x1 expand(2x) → SimpleGate → SCA → 1x1 project → + x
      with LayerNorm-like normalization via GroupNorm.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch * 2, 1, bias=False)
        self.gate = SimpleGate()
        self.sca = SimpleChannelAttention(ch)
        self.conv2 = nn.Conv2d(ch, ch, 1, bias=False)
        # Drop path (optional, disabled by default)
        self.drop_path = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.conv1(x)          # expand 2x
        x = self.gate(x)            # halve back via gating
        x = self.sca(x)             # channel attention
        x = self.conv2(x)           # project back
        x = self.drop_path(x)
        return x + residual


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False)
        self.gn = _gn(ch * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gn(self.conv(x)))


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False)
        self.gn = _gn(ch // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gn(self.conv(x)))


def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    dh = skip.size(2) - x.size(2)
    dw = skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
    return torch.cat([x, skip], dim=1)


class NAFNet(BaseSurrogate):
    """NAFNet: Nonlinear Activation Free Network for dense wind field regression.

    U-shaped encoder-decoder with NAFBlocks (SimpleGate + SCA) instead of
    traditional ReLU/GELU activations. Known for stable training and strong
    performance on image restoration tasks.

    Args:
        depth: number of encoder stages (4, 5, 6, or 7).
        n_c: base channel count; doubles per stage.
        n_blocks: number of NAFBlocks per stage (default 2).
    """

    SUPPORTED_DEPTHS = (4, 5, 6, 7)

    def __init__(self, depth: int = 4, n_c: int = 24, n_blocks: int = 2) -> None:
        super().__init__()
        if depth not in self.SUPPORTED_DEPTHS:
            raise ValueError(f"depth must be in {self.SUPPORTED_DEPTHS}, got {depth}")
        self.depth = depth
        self.n_c = n_c

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, n_c, 3, padding=1, bias=False),
            _gn(n_c),
            nn.ReLU(inplace=True),
        )

        # Encoder stages
        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            blocks = nn.Sequential(*[NAFBlock(ch) for _ in range(n_blocks)])
            self.enc_blocks.append(blocks)
            self.down.append(Downsample(ch))
            ch *= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(*[NAFBlock(ch) for _ in range(n_blocks)])

        # Decoder stages
        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(Upsample(ch))        # ch -> ch//2
            # After up: ch//2, cat with skip (ch//2): total = ch
            # Need to project back to ch//2 for next level
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False),
                _gn(ch // 2),
                *[NAFBlock(ch // 2) for _ in range(n_blocks)],
            ))
            ch //= 2

        # Output head
        self.output_proj = nn.Sequential(
            nn.Conv2d(n_c, n_c, 3, padding=1, bias=False),
            _gn(n_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_c, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        # Encoder
        skips = []
        for blocks, down in zip(self.enc_blocks, self.down):
            x = blocks(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            x = _pad_cat(x, skip)
            x = self.dec_blocks[k](x)

        return self.output_proj(x)
