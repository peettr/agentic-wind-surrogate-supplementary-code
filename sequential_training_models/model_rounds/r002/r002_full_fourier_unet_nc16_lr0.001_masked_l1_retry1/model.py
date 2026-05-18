"""FourierUNet — UNet with Fourier upsampling in decoder.

Replaces standard ConvTranspose2d upsampling with Fourier-based upsampling
that preserves frequency content. Encoder is standard convolution, decoder
uses spectral interpolation for smoother, artifact-free reconstruction.
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


class ConvBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), _gn(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), _gn(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.block(x))


class FourierUpsample(nn.Module):
    """Upsample using Fourier interpolation (zero-pad high frequencies).

    Correctly places low-frequency modes in the standard FFT layout:
    - Rows 0..H//2 get positive vertical frequencies (DC at row 0)
    - Rows H//2..H get negative vertical frequencies (Nyquist at row H-1)
    - Columns 0..W_new//2+1 cover all horizontal frequencies (rfft2 output)
    """

    def __init__(self, ch: int, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(ch, ch // 2, 1, bias=False)
        self.norm = _gn(ch // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # FFT
        x_ft = torch.fft.rfft2(x)  # (B, C, H, W//2+1) complex
        new_h = H * self.scale
        new_w = W * self.scale
        # Allocate zero-padded spectrum
        x_up_ft = torch.zeros(B, C, new_h, new_w // 2 + 1, device=x.device, dtype=x_ft.dtype)
        # Positive vertical frequencies: rows 0..(H//2) map to rows 0..(H//2)
        pos_rows = H // 2 + 1  # includes DC and positive freqs up to Nyquist
        x_up_ft[:, :, :pos_rows, :x_ft.shape[-1]] = x_ft[:, :, :pos_rows, :] * (self.scale ** 2)
        # Negative vertical frequencies: rows (H-pos_rows+1)..(H-1) map to rows (new_h-pos_rows+1)..(new_h-1)
        if H > 1:
            neg_rows = H - pos_rows  # number of negative-frequency rows
            if neg_rows > 0:
                x_up_ft[:, :, new_h - neg_rows:, :x_ft.shape[-1]] = x_ft[:, :, pos_rows:, :] * (self.scale ** 2)
        # iFFT
        x_up = torch.fft.irfft2(x_up_ft, s=(new_h, new_w))
        return F.gelu(self.norm(self.conv(x_up)))


class FourierUNet(BaseSurrogate):
    """UNet with Fourier upsampling for wind field prediction.

    Args:
        n_c: base channel count.
        depth: U-Net depth.
    """

    def __init__(self, n_c: int = 48, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, n_c, 4, stride=2, padding=1, bias=False), _gn(n_c), nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(ConvBlock(ch), ConvBlock(ch)))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            ch *= 2

        self.bottleneck = nn.Sequential(ConvBlock(ch), ConvBlock(ch))

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(FourierUpsample(ch))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False), _gn(ch // 2), nn.GELU(),
                ConvBlock(ch // 2),
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
