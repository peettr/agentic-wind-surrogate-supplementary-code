"""Generated standalone Auto V5 model for dilated_unet.

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
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class DilatedConvBlock(nn.Module):
    """Two dilated Conv3x3 + GN + ReLU layers with configurable dilation."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1) -> None:
        super().__init__()
        pad = dilation  # padding = dilation for same spatial size
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpectralConv2d(nn.Module):
    """Lightweight spectral convolution with bottleneck projection.

    Projects channels down to `hidden` before spectral mixing to keep
    weight tensor small. At 40×40 bottleneck with 4096 channels:
    full mixing = 36GB weights (impossible), but hidden=64 → only 0.5MB.

    Architecture: 1×1 conv (C→hidden) → FFT → weight multiply → iFFT → 1×1 conv (hidden→C)
    """

    def __init__(self, channels: int, modes: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / (hidden * hidden)
        self.proj_down = nn.Conv2d(channels, hidden, 1, bias=False)
        self.proj_up = nn.Conv2d(hidden, channels, 1, bias=False)
        # Complex weight: (hidden, hidden, modes_h, modes_w) — small!
        self.weight = nn.Parameter(
            self.scale * torch.randn(hidden, hidden, modes, modes // 2 + 1, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.proj_down(x)  # (B, hidden, H, W)

        x_ft = torch.fft.rfft2(x, norm="ortho")

        m_h = min(self.modes, x_ft.shape[2])
        m_w = min(self.modes, x_ft.shape[-1])

        out_ft = torch.zeros(x_ft.shape[0], x_ft.shape[1], x_ft.shape[2], x_ft.shape[3],
                             dtype=torch.cfloat, device=x.device)

        x_crop = x_ft[:, :, :m_h, :m_w]       # (B, hidden, m_h, m_w)
        w_crop = self.weight[:, :, :m_h, :m_w] # (hidden, hidden, m_h, m_w)

        # Per-frequency channel mixing in bottleneck space
        out_ft[:, :, :m_h, :m_w] = torch.einsum("ocij,bcij->boij", w_crop, x_crop)

        x = torch.fft.irfft2(out_ft, s=(x.shape[2], x.shape[3]), norm="ortho")
        x = self.proj_up(x)  # (B, channels, H, W)
        return x


def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    dh = skip.size(2) - x.size(2)
    dw = skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
    return torch.cat([x, skip], dim=1)


class DilatedUNet(BaseSurrogate):
    """UNet with dilated convolutions in encoder blocks.

    Args:
        depth: number of encoder stages (5, 6, or 7).
        n_c: base channel count.
        dilation: dilation rate for conv layers (default 2).
        max_ch: optional channel cap for deep stages. Set to None to recover
            the uncapped historical channel schedule.
        spectral_modes: number of frequency modes for bottleneck spectral conv.
            0 = disabled (original DilatedUNet).
    """

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, depth: int = 7, n_c: int = 32, dilation: int = 2,
                 max_ch: int | None = 256,
                 spectral_modes: int = 0) -> None:
        super().__init__()
        if depth not in self.SUPPORTED_DEPTHS:
            raise ValueError(f"depth must be in {self.SUPPORTED_DEPTHS}, got {depth}")
        if max_ch is not None and max_ch < 1:
            raise ValueError(f"max_ch must be positive or None, got {max_ch}")
        self.depth = depth
        self.n_c = n_c
        self.dilation = dilation
        self.max_ch = max_ch
        channel_schedule = [
            min(n_c * 2 ** k, max_ch) if max_ch is not None else n_c * 2 ** k
            for k in range(depth + 1)
        ]

        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch_in = 1
        for ch_out in channel_schedule[:-1]:
            self.enc.append(DilatedConvBlock(ch_in, ch_out, dilation=dilation))
            self.pool.append(nn.MaxPool2d(2))
            ch_in = ch_out

        bottleneck_ch = channel_schedule[-1]
        self.bottleneck = DilatedConvBlock(ch_in, bottleneck_ch, dilation=dilation)

        # Spectral conv side-branch in bottleneck
        self.spectral_modes = spectral_modes
        if spectral_modes > 0:
            self.spectral_conv = SpectralConv2d(bottleneck_ch, modes=spectral_modes)
            self.spectral_gate = nn.Parameter(torch.tensor(0.1))  # learnable gate, starts small

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch_in = bottleneck_ch
        for k in reversed(range(depth)):
            ch_skip = channel_schedule[k]
            self.up.append(nn.ConvTranspose2d(ch_in, ch_skip, 2, stride=2))
            self.dec.append(DilatedConvBlock(ch_skip * 2, ch_skip, dilation=dilation))
            ch_in = ch_skip

        self.head = nn.Sequential(nn.Conv2d(channel_schedule[0], 1, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc_block, pool in zip(self.enc, self.pool):
            x = enc_block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)

        # Add spectral side-branch
        if self.spectral_modes > 0:
            x = x + self.spectral_gate * self.spectral_conv(x)

        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            x = _pad_cat(x, skip)
            x = self.dec[k](x)
        return self.head(x)


class Model(DilatedUNet):
    """Training entrypoint for generated Auto V5 runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)
