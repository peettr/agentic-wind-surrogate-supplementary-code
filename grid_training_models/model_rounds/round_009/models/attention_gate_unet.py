"""Generated standalone Auto V5 model for attention_gate_unet.

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
    """GroupNorm with 8 groups (BatchNorm replacement for EMA compatibility)."""
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class ConvBlock(nn.Module):
    """Two Conv3x3 + GN + ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate: weights skip features by decoder gating signal.

    Args:
        skip_ch:  channels from encoder skip connection.
        gate_ch:  channels from decoder (gating signal, typically same as skip after upsample).
        inter_ch: intermediate channel count for attention computation.
    """

    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int | None = None) -> None:
        super().__init__()
        inter_ch = inter_ch or skip_ch // 2
        if inter_ch == 0:
            inter_ch = 1
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
            _gn(inter_ch),
        )
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            _gn(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            _gn(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted skip features.

        Args:
            skip: encoder features (B, C_skip, H, W).
            gate: decoder gating signal (B, C_gate, H, W) — same spatial size as skip.
        """
        # Align spatial sizes if needed (gate may be slightly different after upsampling)
        if skip.shape[2:] != gate.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)
        a = self.relu(self.W_skip(skip) + self.W_gate(gate))
        a = self.psi(a)
        return skip * a


def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    dh = skip.size(2) - x.size(2)
    dw = skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
    return torch.cat([x, skip], dim=1)


class AttentionGateUNet(BaseSurrogate):
    """UNet with attention gates on every skip connection.

    Args:
        depth: number of encoder stages (5, 6, or 7).
        n_c: base channel count; doubles per stage.

    Defaults target the smoke20 16GB-tier contract at batch=16. The original
    depth=7, n_c=32 default produced roughly 503M parameters and OOMed before
    training.
    """

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, depth: int = 6, n_c: int = 16) -> None:
        super().__init__()
        if depth not in self.SUPPORTED_DEPTHS:
            raise ValueError(f"depth must be in {self.SUPPORTED_DEPTHS}, got {depth}")
        self.depth = depth
        self.n_c = n_c

        # Encoder
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch_in = 1
        for k in range(depth):
            ch_out = n_c * 2 ** k
            self.enc.append(ConvBlock(ch_in, ch_out))
            self.pool.append(nn.MaxPool2d(2))
            ch_in = ch_out

        # Bottleneck
        bottleneck_ch = n_c * 2 ** depth
        self.bottleneck = ConvBlock(ch_in, bottleneck_ch)

        # Decoder + attention gates
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.attn = nn.ModuleList()
        ch_in = bottleneck_ch
        for k in reversed(range(depth)):
            ch_skip = n_c * 2 ** k
            self.up.append(nn.ConvTranspose2d(ch_in, ch_skip, 2, stride=2))
            # Attention gate: skip_ch = ch_skip, gate_ch = ch_skip (after up)
            self.attn.append(AttentionGate(skip_ch=ch_skip, gate_ch=ch_skip))
            self.dec.append(ConvBlock(ch_skip * 2, ch_skip))
            ch_in = ch_skip

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(n_c, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        for enc_block, pool in zip(self.enc, self.pool):
            x = enc_block(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with attention gates
        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            # Apply attention gate
            skip = self.attn[k](skip, x)
            x = _pad_cat(x, skip)
            x = self.dec[k](x)

        return self.head(x)


class Model(AttentionGateUNet):
    """Training entrypoint for generated Auto V5 runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)
