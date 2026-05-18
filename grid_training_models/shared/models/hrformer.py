"""HRFormer — Multi-Resolution Transformer for dense field regression.

Maintains parallel streams at multiple resolutions with cross-resolution
attention, similar to HRNet but using window-based self-attention instead
of convolutions.

Based on: Yuan et al., 2022 (HRFormer)
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


class WindowSelfAttention(nn.Module):
    """Window-based self-attention for efficient local attention."""

    def __init__(self, dim: int, n_heads: int = 4, window_size: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = min(self.window_size, H, W)
        # Pad to multiple of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h])

        _, _, Hp, Wp = x.shape
        nH, nW = Hp // ws, Wp // ws

        # Reshape into windows: (B, nH*nW, ws*ws, C)
        x = x.reshape(B, C, nH, ws, nW, ws).permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B * nH * nW, ws * ws, C)

        # QKV attention
        qkv = self.qkv(x).reshape(B * nH * nW, ws * ws, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*nW*nH, n_heads, ws*ws, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * nH * nW, ws * ws, -1)
        out = self.proj(out)

        # Reshape back
        out = out.reshape(B, nH, nW, ws, ws, -1).permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(B, -1, Hp, Wp)
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out


class HRFormerBlock(nn.Module):
    """HRFormer block: window attention + FFN."""

    def __init__(self, dim: int, n_heads: int = 4, window_size: int = 8) -> None:
        super().__init__()
        self.norm1 = _gn(dim)
        self.attn = WindowSelfAttention(dim, n_heads, window_size)
        self.norm2 = _gn(dim)
        self.ff = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class HRFormer(BaseSurrogate):
    """HRFormer: Multi-Resolution Transformer for wind field prediction.

    U-shaped encoder-decoder with window-based attention at each level.

    Args:
        dim: base channel dimension.
        depth: number of encoder stages.
        n_heads: number of attention heads.
        n_blocks: number of HRFormer blocks per stage.
        window_size: window size for local attention.
    """

    def __init__(self, dim: int = 64, depth: int = 5, n_heads: int = 4,
                 n_blocks: int = 2, window_size: int = 8) -> None:
        super().__init__()
        self.depth = depth

        # Input stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, dim, 4, stride=2, padding=1, bias=False),
            _gn(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            _gn(dim), nn.GELU(),
        )

        # Encoder
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = dim
        for _ in range(depth):
            self.enc.append(nn.Sequential(*[
                HRFormerBlock(ch, n_heads, window_size) for _ in range(n_blocks)
            ]))
            self.down.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False))
            ch *= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            HRFormerBlock(ch, n_heads * 2, max(window_size // 2, 4)) for _ in range(n_blocks)
        ])

        # Decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False))
            self.dec.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 3, padding=1, bias=False),
                _gn(ch // 2), nn.GELU(),
                *[HRFormerBlock(ch // 2, n_heads, window_size) for _ in range(n_blocks)],
            ))
            ch //= 2

        # Output
        self.head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, stride=2, bias=False),
            _gn(dim), nn.GELU(),
            nn.Conv2d(dim, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.stem(x)
        for enc, down in zip(self.enc, self.down):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            dh = skip.size(2) - x.size(2)
            dw = skip.size(3) - x.size(3)
            if dh or dw:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.dec[k](x)
        return self.head(x)
