#!/usr/bin/env python3
"""U-FNO (U-shaped Fourier Neural Operator) conforming to BaseSurrogate contract.

Combines FNO's global spectral mixing with UNet's local edge fidelity.
Architecture: UNet-style encoder-decoder with FactorizedSpectralConv at bottleneck,
preserving skip connections for sharp building edge reconstruction.

Reference: Wen et al., 2022 "U-FNO—An enhanced Fourier neural operator-based 
deep neural network for multiphase flow"

Input/output contract: (B, 1, 640, 640) -> (B, 1, 640, 640) with ReLU output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSurrogate


class FactorizedSpectralConv2d(nn.Module):
    """Factorized 2D spectral convolution (row + column 1D FFTs)."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w_row = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, dtype=torch.cfloat)
        )
        self.w_col = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Row-wise
        x_row = torch.fft.rfft(x, dim=-2, norm="ortho")
        row_m = min(self.modes1, H // 2 + 1)
        out_row = torch.zeros(B, self.out_ch, H // 2 + 1, W, dtype=torch.cfloat, device=x.device)
        # Flatten for einsum
        out_row[:, :, :row_m, :] = torch.einsum(
            "bir,ior->bor",
            x_row[:, :, :row_m, :].reshape(B, C, -1),
            self.w_row[:, :, :row_m].unsqueeze(-1).expand(-1, -1, -1, W).reshape(self.in_ch, self.out_ch, -1),
        ).reshape(B, self.out_ch, row_m, W)
        row_out = torch.fft.irfft(out_row, n=H, dim=-2, norm="ortho")

        # Column-wise
        x_col = torch.fft.rfft(x, dim=-1, norm="ortho")
        col_m = min(self.modes2, W // 2 + 1)
        out_col = torch.zeros(B, self.out_ch, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_col[:, :, :, :col_m] = torch.einsum(
            "bic,ioc->boc",
            x_col[:, :, :, :col_m].reshape(B, C, -1),
            self.w_col[:, :, :col_m].unsqueeze(-2).expand(-1, -1, H, -1).reshape(self.in_ch, self.out_ch, -1),
        ).reshape(B, self.out_ch, H, col_m)
        col_out = torch.fft.irfft(out_col, n=W, dim=-1, norm="ortho")

        return row_out + col_out


class ConvBlock(nn.Module):
    """Double conv: Conv-GN-ReLU x2."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FNOBottleneck(nn.Module):
    """FNO block at bottleneck: spectral conv + pointwise + BN + activation."""
    def __init__(self, width: int, modes: int, n_layers: int = 4) -> None:
        super().__init__()
        modes2 = modes // 2 + 1
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "spectral": FactorizedSpectralConv2d(width, width, modes, modes2),
                "skip": nn.Conv2d(width, width, 1),
                "gn": nn.GroupNorm(min(32, width), width),
            }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.gelu(layer["gn"](layer["spectral"](x) + layer["skip"](x)))
        return x


class UFNO(BaseSurrogate):
    """U-shaped Fourier Neural Operator (U-FNO).

    UNet encoder-decoder with FNO spectral blocks at the bottleneck.
    Combines global receptive field (via FFT) with local edge fidelity (via skip connections).

    Args:
        n_c: base channel width (default 32). Total params ~55M at n_c=32.
        depth: number of encoder/decoder stages (default 5, gives 5-level UNet).
        modes: FFT modes per dimension at bottleneck (default 32).
        fno_layers: number of FNO blocks at bottleneck (default 4).
        activation: 'gelu' or 'relu' (default 'gelu').
        training: dict of training extras — ignored by model.
    """

    def __init__(
        self,
        n_c: int = 32,
        depth: int = 5,
        modes: int = 32,
        fno_layers: int = 4,
        activation: str = "gelu",
        training: dict | None = None,
    ) -> None:
        super().__init__()
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = 1
        for i in range(depth):
            ch_out = n_c * (2 ** i)
            self.encoders.append(ConvBlock(ch_in, ch_out))
            self.pools.append(nn.Conv2d(ch_out, ch_out, 2, stride=2))
            ch_in = ch_out

        # Bottleneck = FNO
        bottleneck_ch = n_c * (2 ** depth)
        self.bottleneck_conv = ConvBlock(ch_in, bottleneck_ch)
        self.fno = FNOBottleneck(bottleneck_ch, modes, fno_layers)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            ch_skip = n_c * (2 ** i)
            ch_up = bottleneck_ch if i == depth - 1 else n_c * (2 ** (i + 1))
            self.upconvs.append(nn.ConvTranspose2d(ch_up, ch_skip, 2, stride=2))
            self.decoders.append(ConvBlock(ch_skip * 2, ch_skip))  # *2 for skip concat

        # Output
        self.out_conv = nn.Sequential(
            nn.Conv2d(n_c, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck_conv(x)
        x = self.fno(x)

        # Decoder
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.out_conv(x)


if __name__ == "__main__":
    for n_c in [16, 32]:
        for depth in [4, 5, 6]:
            m = UFNO(n_c=n_c, depth=depth)
            n_params = sum(p.numel() for p in m.parameters())
            x = torch.randn(2, 1, 640, 640)
            with torch.no_grad():
                y = m(x)
            print(f"UFNO n_c={n_c} depth={depth}: params={n_params:,} ({n_params/1e6:.1f}M) out={tuple(y.shape)} min={y.min():.4f}")
