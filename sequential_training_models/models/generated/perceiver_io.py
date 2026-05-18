"""Perceiver IO - latent-space attention for dense regression."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


def _conv_reflect(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                  bias: bool = False) -> nn.Sequential:
    pad = kernel_size // 2
    return nn.Sequential(
        nn.ReflectionPad2d(pad),
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=bias),
    )


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
        b, nq, _ = q.shape
        _, nkv, _ = kv.shape
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        q = self.q_proj(q).reshape(b, nq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).reshape(b, nkv, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).reshape(b, nkv, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, nq, -1)
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


class perceiver_io(nn.Module):
    """Perceiver IO for dense wind field regression."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 n_c: int = 16, depth: int = 7) -> None:
        super().__init__()
        latent_dim = n_c
        n_latents = 1024
        n_heads = 4
        enc_dim = 128

        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.latent_side = int(math.isqrt(n_latents))

        self.encoder = nn.Sequential(
            _conv_reflect(in_channels, enc_dim, 4, stride=2, bias=False),
            _gn(enc_dim), nn.GELU(),
            _conv_reflect(enc_dim, enc_dim, 4, stride=2, bias=False),
            _gn(enc_dim), nn.GELU(),
            _conv_reflect(enc_dim, enc_dim, 3, stride=2, bias=False),
            _gn(enc_dim), nn.GELU(),
        )

        self.latents = nn.Parameter(torch.randn(1, n_latents, latent_dim) * 0.02)

        self.blocks = nn.ModuleList([
            PerceiverBlock(latent_dim, enc_dim, n_heads) for _ in range(depth)
        ])

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, enc_dim),
            nn.GELU(),
        )

        self.upsample = nn.Sequential(
            _conv_reflect(enc_dim, enc_dim, 3, bias=False),
            _gn(enc_dim), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv_reflect(enc_dim, enc_dim, 3, bias=False),
            _gn(enc_dim), nn.GELU(),
            nn.Upsample(scale_factor=5, mode="bilinear", align_corners=False),
            _conv_reflect(enc_dim, enc_dim, 3, bias=False),
            _gn(enc_dim), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(enc_dim, out_channels, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(x)
        x = torch.where(mask, x, torch.zeros_like(x))

        b = x.size(0)
        orig_h, orig_w = x.shape[2], x.shape[3]

        enc = self.encoder(x)
        enc_flat = enc.reshape(b, enc.size(1), -1).permute(0, 2, 1)

        latent = self.latents.expand(b, -1, -1)

        for block in self.blocks:
            latent = block(latent, enc_flat)

        latent_proj = self.latent_proj(latent)
        latent_2d = latent_proj.permute(0, 2, 1).reshape(
            b, enc.size(1), self.latent_side, self.latent_side
        )

        out = self.upsample(latent_2d)

        if out.shape[2] != orig_h or out.shape[3] != orig_w:
            out = F.interpolate(out, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

        return out