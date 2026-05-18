from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.assign = nn.Linear(dim, n_slices)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        residual = x
        x = self.norm(x)
        logits = self.assign(x)
        pool = logits.softmax(dim=1)
        dispatch = logits.softmax(dim=-1)
        slices = torch.einsum("bns,bnd->bsd", pool, x)

        qkv = self.qkv(slices).reshape(b, self.n_slices, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        slices = (attn @ v).transpose(1, 2).reshape(b, self.n_slices, d)
        x = torch.einsum("bns,bsd->bnd", dispatch, slices)
        return residual + self.proj(x)


class TransolverBlock(nn.Module):
    def __init__(self, dim: int, n_slices: int = 8, n_heads: int = 4) -> None:
        super().__init__()
        self.slice_attn = SliceAttention(dim, n_slices, n_heads)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.slice_attn(x)
        x = x + self.ff(x)
        return x


class transolver_lite(nn.Module):
    """Lightweight Transolver for wind pressure prediction."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, n_c: int = 16, depth: int = 7) -> None:
        super().__init__()
        dim = n_c
        patch_size = 8
        n_slices = 8
        n_heads = 4

        self.patch_size = patch_size
        self.dim = dim

        self.patch_embed = nn.Sequential(
            nn.ReflectionPad2d(patch_size // 2),
            nn.Conv2d(in_channels, dim, patch_size, stride=patch_size, bias=False),
            _gn(dim),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(*[
            TransolverBlock(dim, n_slices, n_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        self.unpatch = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            _gn(dim),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim // 2, 3, bias=False),
            _gn(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(x)
        x = torch.where(valid, x, torch.zeros_like(x))

        b, _, h, w = x.shape

        x = self.patch_embed(x)
        _, d, hp, wp = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(b, d, hp, wp)
        x = self.unpatch(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        x = F.relu(self.output_proj(x))

        if valid.shape[1] == x.shape[1]:
            x = torch.where(valid, x, torch.zeros_like(x))

        return x



