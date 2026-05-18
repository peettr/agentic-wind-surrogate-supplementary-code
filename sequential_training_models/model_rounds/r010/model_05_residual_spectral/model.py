"""Residual spectral network for wind pressure prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    g = next(x for x in range(min(32, ch), 0, -1) if ch % x == 0)
    return nn.GroupNorm(g, ch)


class ReflectConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, bias: bool = False) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ReflectConv2d(ch, ch, 3, bias=False),
            _gn(ch),
            nn.GELU(),
            ReflectConv2d(ch, ch, 3, bias=False),
            _gn(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.block(x))


class SpectralBottleneck(nn.Module):
    def __init__(self, ch: int, modes: int = 8) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (ch * ch)
        self.w_real = nn.Parameter(scale * torch.randn(ch, ch, modes, modes))
        self.w_imag = nn.Parameter(scale * torch.randn(ch, ch, modes, modes))
        self.local = nn.Conv2d(ch, ch, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch * 2, ch, 1, bias=False),
        )
        self.norm = _gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        x_ft = torch.fft.rfft2(x)
        m1 = min(self.modes, x.shape[2])
        m2 = min(self.modes, x_ft.shape[-1])
        w_complex = torch.complex(
            self.w_real[:, :, :m1, :m2],
            self.w_imag[:, :, :m1, :m2],
        )

        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m1, :m2],
            w_complex,
        )
        x_spec = torch.fft.irfft2(out_ft, s=x.shape[2:])

        x = self.local(x) + x_spec
        x = x + self.mlp(x)
        return x + residual


class residual_spectral(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_c: int = 16,
        depth: int = 7,
    ) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            ReflectConv2d(in_channels, n_c, 3, bias=False),
            _gn(n_c),
            nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(ResBlock(ch), ResBlock(ch)))
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False),
                    _gn(ch * 2),
                    nn.GELU(),
                )
            )
            ch *= 2

        self.bottleneck = nn.Sequential(
            SpectralBottleneck(ch),
            SpectralBottleneck(ch),
        )

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False),
                    _gn(ch // 2),
                    nn.GELU(),
                )
            )
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch // 2, 1, bias=False),
                    _gn(ch // 2),
                    nn.GELU(),
                    ResBlock(ch // 2),
                )
            )
            ch //= 2

        self.output_proj = nn.Sequential(
            ReflectConv2d(n_c, n_c, 3, bias=False),
            _gn(n_c),
            nn.GELU(),
            nn.Conv2d(n_c, out_channels, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(x)
        x = torch.where(valid, x, torch.zeros_like(x))

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
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2], mode="reflect")
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[k](x)

        return self.output_proj(x)


