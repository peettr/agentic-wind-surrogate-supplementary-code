"""FNO-2d â€” Fourier Neural Operator for 2D dense regression.

Applies spectral convolution layers in frequency domain, followed by
pointwise MLP layers. Known for resolution-invariant learning and strong
performance on PDE surrogate tasks.

Based on: Li et al., 2021 (Fourier Neural Operator for Parametric PDEs)
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


class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT â†’ linear transform on modes â†’ iFFT."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weight1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.weight2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (B, in_ch, m1, m2) Ã— (in_ch, out_ch, m1, m2) â†’ (B, out_ch, m1, m2)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, self.weight1.shape[1], H, x_ft.shape[-1],
                             device=x.device, dtype=torch.cfloat)
        m1 = min(self.modes1, H)
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weight1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weight2[:, :, :m1, :m2])

        return torch.fft.irfft2(out_ft, s=(H, W))


class FNOBlock(nn.Module):
    """FNO layer: spectral conv + pointwise MLP + residual."""

    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.mlp = nn.Sequential(
            nn.Conv2d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, 1),
        )
        self.norm = _gn(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.spectral(x) + self.mlp(x)
        return x + residual


class FNO2d(BaseSurrogate):
    """Fourier Neural Operator for 2D wind field prediction.

    U-shaped encoder-decoder with FNO spectral layers at each resolution.

    Args:
        width: channel width for FNO layers.
        modes: number of Fourier modes to keep (per dimension).
        depth: number of U-Net encoder/decoder stages.
        n_blocks: number of FNO blocks per stage.
    """

    def __init__(self, width: int = 48, modes: int = 16, depth: int = 4,
                 n_blocks: int = 2) -> None:
        super().__init__()
        self.depth = depth

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False),
            _gn(width), nn.GELU(),
        )

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = width
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(*[FNOBlock(ch, modes, modes) for _ in range(n_blocks)]))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            ch *= 2
            modes = max(modes // 2, 4)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[FNOBlock(ch, modes, modes) for _ in range(n_blocks)])

        # Decoder
        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False), _gn(ch // 2), nn.GELU()))
            ch //= 2
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch * 2, ch, 1, bias=False), _gn(ch), nn.GELU(),
                *[FNOBlock(ch, modes, modes) for _ in range(n_blocks)],
            ))
            modes = min(modes * 2, 16)

        # Output
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            _gn(width), nn.GELU(),
            nn.Conv2d(width, 1, 1),
            nn.ReLU(),
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
            # Pad if needed
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh != 0 or dw != 0:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[k](x)
        return self.output_proj(x)



