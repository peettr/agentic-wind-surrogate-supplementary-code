"""ResidualSpectralNet â€” Deep residual network with spectral bottleneck.

Deep residual encoder-decoder with a spectral (Fourier) processing bottleneck.
The encoder/decoder use standard residual blocks, while the bottleneck
leverages Fourier spectral convolutions for global information mixing.
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


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), _gn(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), _gn(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.block(x))


class SpectralBottleneck(nn.Module):
    """Spectral processing at the deepest level."""

    def __init__(self, ch: int, modes: int = 8) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (ch * ch)
        self.w_real = nn.Parameter(scale * torch.randn(ch, ch, modes, modes))
        self.w_imag = nn.Parameter(scale * torch.randn(ch, ch, modes, modes))
        self.local = nn.Conv2d(ch, ch, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 1, bias=False), nn.GELU(),
            nn.Conv2d(ch * 2, ch, 1, bias=False),
        )
        self.norm = _gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        # Spectral conv
        x_ft = torch.fft.rfft2(x)
        m1 = min(self.modes, x.shape[2])
        m2 = min(self.modes, x_ft.shape[-1])
        w_complex = torch.complex(self.w_real[:, :, :m1, :m2], self.w_imag[:, :, :m1, :m2])
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], w_complex)
        x_spec = torch.fft.irfft2(out_ft, s=x.shape[2:])
        # Combine
        x = self.local(x) + x_spec
        x = x + self.mlp(x)
        return x + residual


class ResidualSpectralNet(BaseSurrogate):
    """Deep residual + spectral bottleneck for wind field prediction.

    Args:
        n_c: base channel count.
        depth: U-Net depth.
        modes: Fourier modes in bottleneck.
    """

    def __init__(self, n_c: int = 48, depth: int = 4, modes: int = 8) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, n_c, 3, padding=1, bias=False), _gn(n_c), nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(ResBlock(ch), ResBlock(ch)))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            ch *= 2

        self.bottleneck = nn.Sequential(
            SpectralBottleneck(ch, modes),
            SpectralBottleneck(ch, modes),
        )

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False), _gn(ch // 2), nn.GELU()))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False), _gn(ch // 2), nn.GELU(),
                ResBlock(ch // 2),
            ))
            ch //= 2

        self.output_proj = nn.Sequential(
            nn.Conv2d(n_c, n_c, 3, padding=1, bias=False), _gn(n_c), nn.GELU(),
            nn.Conv2d(n_c, 1, 1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.output_proj(x)



