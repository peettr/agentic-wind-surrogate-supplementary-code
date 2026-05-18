"""Factorized Fourier Neural Operator (F-FNO) for wind pressure prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class FactorizedSpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / max(1, in_ch * out_ch)
        self.w_row = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, dtype=torch.cfloat)
        )
        self.w_col = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _complex_mul_1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        x_row_ft = torch.fft.rfft(x, dim=-2, norm="ortho")
        row_modes = min(self.modes1, h // 2 + 1)
        out_row_ft = torch.zeros(
            b,
            self.out_ch,
            h // 2 + 1,
            w,
            dtype=torch.cfloat,
            device=x.device,
        )
        row_weight = (
            self.w_row[:, :, :row_modes]
            .unsqueeze(-1)
            .expand(-1, -1, -1, w)
            .reshape(self.in_ch, self.out_ch, row_modes * w)
        )
        out_row_ft[:, :, :row_modes, :] = self._complex_mul_1d(
            x_row_ft[:, :, :row_modes, :].reshape(b, c, row_modes * w),
            row_weight,
        ).reshape(b, self.out_ch, row_modes, w)
        row_out = torch.fft.irfft(out_row_ft, n=h, dim=-2, norm="ortho")

        x_col_ft = torch.fft.rfft(x, dim=-1, norm="ortho")
        col_modes = min(self.modes2, w // 2 + 1)
        out_col_ft = torch.zeros(
            b,
            self.out_ch,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        col_weight = (
            self.w_col[:, :, :col_modes]
            .unsqueeze(-2)
            .expand(-1, -1, h, -1)
            .reshape(self.in_ch, self.out_ch, h * col_modes)
        )
        out_col_ft[:, :, :, :col_modes] = self._complex_mul_1d(
            x_col_ft[:, :, :, :col_modes].reshape(b, c, h * col_modes),
            col_weight,
        ).reshape(b, self.out_ch, h, col_modes)
        col_out = torch.fft.irfft(out_col_ft, n=w, dim=-1, norm="ortho")

        return row_out + col_out


class FFNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = FactorizedSpectralConv2d(width, width, modes1, modes2)
        self.skip = nn.Conv2d(width, width, kernel_size=1)
        self.norm = nn.GroupNorm(1, width)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.skip(x)))


class ffno(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_c: int = 16,
        depth: int = 7,
    ) -> None:
        super().__init__()
        modes = 32
        modes1 = modes
        modes2 = modes

        self.pad = 8
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, n_c, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            FFNOBlock(n_c, modes1, modes2) for _ in range(depth)
        )
        self.project = nn.Sequential(
            nn.Conv2d(n_c, n_c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_c, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(x)
        x = torch.where(nan_mask, torch.zeros_like(x), x)

        if self.pad > 0:
            x = nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")

        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = self.project(x)

        if self.pad > 0:
            x = x[..., self.pad : -self.pad, self.pad : -self.pad]

        return x


