"""Generated standalone Grid model for dilated_hrformer.

This generated file is the training source of truth for this run.
Runtime model construction is local to this file rather than registry delegation.
"""
from __future__ import annotations

import math
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


class WindowAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, window_size: int = 8) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws = self.window_size
        # Pad to window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])
        _, _, Hp, Wp = x.shape
        # Reshape to windows
        x = x.reshape(B, C, Hp // ws, ws, Wp // ws, ws).permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B * (Hp // ws) * (Wp // ws), ws * ws, C)
        # Attention
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        x = residual + self.proj(x)
        # Reshape back
        x = x.reshape(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x


class DilatedAttentionBlock(nn.Module):
    def __init__(self, ch: int, n_heads: int = 4, window_size: int = 8) -> None:
        super().__init__()
        self.attn = WindowAttention(ch, n_heads, window_size)
        self.dilated = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, dilation=1, bias=False), _gn(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2, bias=False), _gn(ch),
        )
        self.norm = _gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + F.gelu(self.dilated(x))
        return self.norm(x)


class DilatedHRFormer(BaseSurrogate):
    """Window attention + dilated convolution hybrid for wind field prediction.

    Args:
        n_c: base channel count.
        depth: U-Net depth.
        n_heads: attention heads.
        window_size: attention window size.
    """

    def __init__(self, n_c: int = 48, depth: int = 4, n_heads: int = 4,
                 window_size: int = 8) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, n_c, 4, stride=2, padding=1, bias=False), _gn(n_c), nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(
                DilatedAttentionBlock(ch, n_heads, window_size),
                DilatedAttentionBlock(ch, n_heads, window_size),
            ))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            ch *= 2

        self.bottleneck = nn.Sequential(
            DilatedAttentionBlock(ch, n_heads, window_size),
            DilatedAttentionBlock(ch, n_heads, window_size),
        )

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False), _gn(ch // 2), nn.GELU()))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False), _gn(ch // 2), nn.GELU(),
                DilatedAttentionBlock(ch // 2, n_heads, window_size),
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


class Model(DilatedHRFormer):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



