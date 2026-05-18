from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )
        self.w2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _complex_mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            b,
            self.out_ch,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, x_ft.size(-2))
        m2 = min(self.modes2, x_ft.size(-1))

        out_ft[:, :, :m1, :m2] = self._complex_mul(
            x_ft[:, :, :m1, :m2],
            self.w1[:, :, :m1, :m2],
        )
        out_ft[:, :, -m1:, :m2] = self._complex_mul(
            x_ft[:, :, -m1:, :m2],
            self.w2[:, :, :m1, :m2],
        )

        return torch.fft.irfft2(out_ft, s=x.shape[-2:], norm="ortho")


class ReflectConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.skip = ReflectConv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.skip(x))


class fno_v3(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_c: int = 16,
        depth: int = 7,
    ) -> None:
        super().__init__()
        modes = 12
        width = n_c

        self.lift = ReflectConv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock(width, modes) for _ in range(depth))
        self.project = nn.Sequential(
            ReflectConv2d(width, width, kernel_size=1),
            nn.GELU(),
            ReflectConv2d(width, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(x)
        x = torch.where(valid, x, torch.zeros_like(x))

        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = self.project(x)

        return x


