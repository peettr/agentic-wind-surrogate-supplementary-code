"""Perceiver IO — latent-space attention for dense regression.

Compresses the 640x640 input into a small latent array, applies multiple rounds
of cross-attention and self-attention, then decodes back to full resolution.
Efficient for large spatial domains.

Based on: Jaegle et al., 2021 (Perceiver IO)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSurrogate


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class CrossAttention(nn.Module):
    """Cross-attention between queries and key-value pairs."""

    def __init__(self, q_dim: int, kv_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = q_dim // n_heads
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.out = nn.Linear(q_dim, q_dim)
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nq, _ = q.shape
        _, Nkv, _ = kv.shape
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        Q = self.q_proj(q).reshape(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).reshape(B, Nkv, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).reshape(B, Nkv, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, Nq, -1)
        return self.out(out)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = CrossAttention(dim, dim, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x)


class PerceiverBlock(nn.Module):
    """One Perceiver block: cross-attention + self-attention + FFN."""

    def __init__(self, latent_dim: int, input_dim: int, n_heads: int = 4,
                 ff_mult: int = 4) -> None:
        super().__init__()
        self.cross_attn = CrossAttention(latent_dim, input_dim, n_heads)
        self.self_attn = SelfAttention(latent_dim, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * ff_mult),
            nn.GELU(),
            nn.Linear(latent_dim * ff_mult, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, latent: torch.Tensor, input_arr: torch.Tensor) -> torch.Tensor:
        latent = latent + self.cross_attn(latent, input_arr)
        latent = latent + self.self_attn(self.norm(latent))
        latent = latent + self.ff(self.norm(latent))
        return latent


class PerceiverIO(BaseSurrogate):
    """Perceiver IO for dense wind field regression.

    Encodes input to latent space, applies multiple attention rounds,
    decodes back to full resolution.

    Args:
        latent_dim: dimension of latent arrays.
        n_latents: number of latent vectors.
        n_blocks: number of Perceiver blocks.
        n_heads: number of attention heads.
        enc_dim: encoder output channel dimension.
    """

    def __init__(self, latent_dim: int = 256, n_latents: int = 1024,
                 n_blocks: int = 6, n_heads: int = 4, enc_dim: int = 128) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.latent_side = int(math.isqrt(n_latents))  # 32 for 1024

        # Input encoder: conv to (B, enc_dim, H/8, W/8)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, enc_dim, 4, stride=2, padding=1, bias=False),
            _gn(enc_dim), nn.GELU(),
            nn.Conv2d(enc_dim, enc_dim, 4, stride=2, padding=1, bias=False),
            _gn(enc_dim), nn.GELU(),
            nn.Conv2d(enc_dim, enc_dim, 3, stride=2, padding=1, bias=False),
            _gn(enc_dim), nn.GELU(),
        )

        # Learnable latent arrays
        self.latents = nn.Parameter(torch.randn(1, n_latents, latent_dim) * 0.02)

        # Perceiver blocks
        self.blocks = nn.ModuleList([
            PerceiverBlock(latent_dim, enc_dim, n_heads) for _ in range(n_blocks)
        ])

        # Output decoder: project latent_dim -> enc_dim, then reshape + upsample
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, enc_dim),
            nn.GELU(),
        )

        # Progressive upsampling: 32x32 -> 640x640
        self.upsample = nn.Sequential(
            nn.Conv2d(enc_dim, enc_dim, 3, padding=1, bias=False),
            _gn(enc_dim), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64
            nn.Conv2d(enc_dim, enc_dim, 3, padding=1, bias=False),
            _gn(enc_dim), nn.GELU(),
            nn.Upsample(scale_factor=5, mode='bilinear', align_corners=False),  # 320
            nn.Conv2d(enc_dim, enc_dim, 3, padding=1, bias=False),
            _gn(enc_dim), nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 640
            nn.Conv2d(enc_dim, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Encode input
        enc = self.encoder(x)  # (B, enc_dim, H/8, W/8)
        enc_flat = enc.reshape(B, enc.size(1), -1).permute(0, 2, 1)  # (B, N, enc_dim)

        # Expand latents
        latent = self.latents.expand(B, -1, -1)  # (B, n_latents, latent_dim)

        # Apply Perceiver blocks
        for block in self.blocks:
            latent = block(latent, enc_flat)

        # Project latent -> enc_dim channels
        latent_proj = self.latent_proj(latent)  # (B, n_latents, enc_dim)

        # Reshape to 2D grid (n_latents = latent_side²)
        latent_2d = latent_proj.permute(0, 2, 1).reshape(B, enc.size(1), self.latent_side, self.latent_side)

        # Progressive upsampling to full resolution
        out = self.upsample(latent_2d)

        # Ensure exact output size
        if out.shape[2] != orig_h or out.shape[3] != orig_w:
            out = F.interpolate(out, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return out
