"""Generated standalone Auto V5 model for transolver_lite.

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


class SliceAttention(nn.Module):
    """Lightweight slice-based attention."""

    def __init__(self, dim: int, n_slices: int = 8, n_heads: int = 4) -> None:
        super().__init__()
        self.n_slices = n_slices
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return residual + self.proj(x)


class TransolverBlock(nn.Module):
    def __init__(self, dim: int, n_slices: int = 8, n_heads: int = 4) -> None:
        super().__init__()
        self.slice_attn = SliceAttention(dim, n_slices, n_heads)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.slice_attn(x)
        x = x + self.ff(x)
        return x


class TransolverLite(BaseSurrogate):
    """Lightweight Transolver for wind field prediction.

    Args:
        dim: feature dimension.
        depth: number of Transolver blocks.
        n_slices: number of physics slices.
        n_heads: attention heads.
        patch_size: spatial patch size for tokenization.
    """

    def __init__(self, dim: int = 128, depth: int = 4, n_slices: int = 8,
                 n_heads: int = 4, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, dim, patch_size, stride=patch_size, bias=False),
            _gn(dim), nn.GELU(),
        )

        # Transolver blocks
        self.blocks = nn.Sequential(*[
            TransolverBlock(dim, n_slices, n_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # Unpatch
        self.unpatch = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            _gn(dim), nn.GELU(),
        )
        
        # Output projection to 1 channel
        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            _gn(dim // 2), nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        ps = self.patch_size
        x = self.patch_embed(x)  # (B, dim, H/ps, W/ps)
        B, D, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, h*w, dim)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, D, h, w)
        x = self.unpatch(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return F.relu(self.output_proj(x))


class Model(TransolverLite):
    """Training entrypoint for generated Auto V5 runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)
