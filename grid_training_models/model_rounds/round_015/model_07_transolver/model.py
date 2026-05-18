"""Generated standalone Grid model for transolver.

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


class PhysicsAttention(nn.Module):
    """Attention over learned physics slices (token groups)."""

    def __init__(self, dim: int, n_slices: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.n_slices = n_slices
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Slice assignment: predicts soft assignment of spatial points to slices
        self.slice_proj = nn.Linear(dim, n_slices)
        # Slice-level query/key/value
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        # Reshape to (B, N, C)
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)

        residual = x_flat
        x_norm = self.norm(x_flat)

        # Compute slice assignments
        attn_logits = self.slice_proj(x_norm)  # (B, N, n_slices)
        slice_weights = F.softmax(attn_logits, dim=1)  # normalize over spatial

        # Aggregate to slice-level: weighted average
        slice_repr = torch.einsum('bnc,bns->bsc', x_norm, slice_weights)  # (B, S, C)

        # Self-attention on slice-level
        Q = self.q(slice_repr).reshape(B, self.n_slices, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k(slice_repr).reshape(B, self.n_slices, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v(slice_repr).reshape(B, self.n_slices, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        slice_out = (attn @ V).transpose(1, 2).reshape(B, self.n_slices, C)

        # Distribute back to spatial points
        out = torch.einsum('bsc,bns->bnc', slice_out, slice_weights)
        out = self.out(out) + residual

        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class TransolverBlock(nn.Module):
    """Transolver block: PhysicsAttention + FFN."""

    def __init__(self, dim: int, n_slices: int = 32, n_heads: int = 4,
                 ff_mult: int = 4) -> None:
        super().__init__()
        self.attn = PhysicsAttention(dim, n_slices, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.attn(x)
        # FFN
        x_flat = x.permute(0, 2, 3, 1)
        x_flat = x_flat + self.ff(self.norm2(x_flat))
        return x_flat.permute(0, 3, 1, 2)


class Transolver(BaseSurrogate):
    """Transolver: Physics-Attention Transformer for wind field regression.

    U-shaped encoder-decoder with TransolverBlocks at each resolution.

    Args:
        hidden: base channel dimension.
        depth: number of encoder stages.
        n_slices: number of physics slices for attention.
        n_heads: number of attention heads.
        n_blocks: number of TransolverBlocks per stage.
    """

    def __init__(self, hidden: int = 96, depth: int = 4, n_slices: int = 32,
                 n_heads: int = 4, n_blocks: int = 2) -> None:
        super().__init__()
        self.depth = depth

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Conv2d(1, hidden, 4, stride=2, padding=1, bias=False),
            _gn(hidden),
            nn.GELU(),
        )

        # Encoder
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = hidden
        for _ in range(depth):
            self.enc.append(nn.Sequential(*[
                TransolverBlock(ch, n_slices, n_heads) for _ in range(n_blocks)
            ]))
            self.down.append(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False))
            ch *= 2

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            TransolverBlock(ch, n_slices, n_heads) for _ in range(n_blocks)
        ])

        # Decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False))
            self.dec.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 3, padding=1, bias=False),
                _gn(ch // 2),
                nn.GELU(),
                *[TransolverBlock(ch // 2, n_slices, n_heads) for _ in range(n_blocks)],
            ))
            ch //= 2

        # Output head
        self.head = nn.Sequential(
            nn.ConvTranspose2d(hidden, hidden, 2, stride=2, bias=False),
            _gn(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.input_embed(x)
        for enc, down in zip(self.enc, self.down):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            # Pad if needed
            dh = skip.size(2) - x.size(2)
            dw = skip.size(3) - x.size(3)
            if dh != 0 or dw != 0:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.dec[k](x)
        return self.head(x)


class Model(Transolver):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



