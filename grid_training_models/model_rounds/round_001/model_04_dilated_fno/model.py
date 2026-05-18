"""Generated standalone Grid model for dilated_fno.

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


class SpectralSkip(nn.Module):
    """Spectral convolution for skip connection."""

    def __init__(self, ch: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (ch * ch)
        self.weight = nn.Parameter(scale * torch.rand(ch, ch, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros_like(x_ft)
        h = min(self.modes, x.shape[2])
        out_ft[:, :, :h, :m] = torch.einsum("bixy,ioxy->boxy",
            x_ft[:, :, :h, :m], self.weight[:, :, :h, :m])
        return x + torch.fft.irfft2(out_ft, s=x.shape[2:])


class DilatedConvBlock(nn.Module):
    """Multi-rate dilated convolution block."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=2, dilation=2, bias=False)
        self.conv4 = nn.Conv2d(ch, ch, 3, padding=4, dilation=4, bias=False)
        self.proj = nn.Conv2d(ch * 3, ch, 1, bias=False)
        self.norm = _gn(ch)
        self.gn1 = _gn(ch)
        self.gn2 = _gn(ch)
        self.gn4 = _gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.gelu(self.gn1(self.conv1(x)))
        x2 = F.gelu(self.gn2(self.conv2(x)))
        x4 = F.gelu(self.gn4(self.conv4(x)))
        return x + self.norm(self.proj(torch.cat([x1, x2, x4], dim=1)))


class DilatedFNO(BaseSurrogate):
    """Dilated convolution + FNO spectral skip connections.

    Args:
        n_c: base channel count.
        depth: U-Net depth.
        modes: Fourier modes for spectral skip.
    """

    def __init__(self, n_c: int = 48, depth: int = 4, modes: int = 16) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, n_c, 4, stride=2, padding=1, bias=False), _gn(n_c), nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        self.spectral_skips = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(
                DilatedConvBlock(ch), DilatedConvBlock(ch),
            ))
            self.spectral_skips.append(SpectralSkip(ch, modes))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            ch *= 2
            modes = max(modes // 2, 4)

        self.bottleneck = nn.Sequential(DilatedConvBlock(ch), DilatedConvBlock(ch))

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False), _gn(ch // 2), nn.GELU()))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False), _gn(ch // 2), nn.GELU(),
                DilatedConvBlock(ch // 2),
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
        for blocks, spec, down in zip(self.enc_blocks, self.spectral_skips, self.down):
            x = blocks(x)
            x = spec(x)  # spectral skip processing
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


class Model(DilatedFNO):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



