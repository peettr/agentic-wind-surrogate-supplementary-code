"""Generated standalone Auto V5 model for ffno.

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



class FactorizedSpectralConv2d(nn.Module):
    """Factorized 2D spectral convolution using separate 1D FFTs along rows and columns.

    Instead of a full 2D FFT (O(N^2 * modes1 * modes2) params),
    applies 1D spectral conv along rows then columns (O(N^2 * (modes1 + modes2)) params).
    """

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1  # modes along height (rows)
        self.modes2 = modes2  # modes along width (columns)
        scale = 1.0 / (in_ch * out_ch)

        # Row-wise (height dimension) spectral weights
        # Applied via torch.fft.rfft on dim=-2 (height)
        self.w_row = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, dtype=torch.cfloat)
        )
        # Column-wise (width dimension) spectral weights
        # Applied via torch.fft.rfft on dim=-1 (width)
        self.w_col = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _complex_mul_1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """1D complex multiply: (B, I, m) × (I, O, m) → (B, O, m)"""
        return torch.einsum("bix,iox->box", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # --- Row-wise spectral convolution (along height) ---
        # rfft along height dimension (dim=-2), keep width intact
        x_row_ft = torch.fft.rfft(x, dim=-2, norm="ortho")  # (B, C, H//2+1, W)
        row_modes = min(self.modes1, H // 2 + 1)
        out_row_ft = torch.zeros(
            B, self.out_ch, H // 2 + 1, W,
            dtype=torch.cfloat, device=x.device,
        )
        # Apply spectral weights to low-frequency modes along height
        out_row_ft[:, :, :row_modes, :] = self._complex_mul_1d(
            x_row_ft[:, :, :row_modes, :].reshape(B, C, row_modes * W),
            self.w_row[:, :, :row_modes].unsqueeze(-1).expand(-1, -1, -1, W).reshape(self.in_ch, self.out_ch, row_modes * W),
        ).reshape(B, self.out_ch, row_modes, W)
        row_out = torch.fft.irfft(out_row_ft, n=H, dim=-2, norm="ortho")  # (B, out_ch, H, W)

        # --- Column-wise spectral convolution (along width) ---
        x_col_ft = torch.fft.rfft(x, dim=-1, norm="ortho")  # (B, C, H, W//2+1)
        col_modes = min(self.modes2, W // 2 + 1)
        out_col_ft = torch.zeros(
            B, self.out_ch, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_col_ft[:, :, :, :col_modes] = self._complex_mul_1d(
            x_col_ft[:, :, :, :col_modes].reshape(B, C, H * col_modes),
            self.w_col[:, :, :col_modes].unsqueeze(-2).expand(-1, -1, H, -1).reshape(self.in_ch, self.out_ch, H * col_modes),
        ).reshape(B, self.out_ch, H, col_modes)
        col_out = torch.fft.irfft(out_col_ft, n=W, dim=-1, norm="ortho")  # (B, out_ch, H, W)

        # Combine row and column outputs
        return row_out + col_out


class FFNOBlock(nn.Module):
    """Single F-FNO block: FactorizedSpectralConv + pointwise Conv + BatchNorm + activation."""

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.spectral = FactorizedSpectralConv2d(width, width, modes1, modes2)
        self.skip = nn.Conv2d(width, width, kernel_size=1)
        self.bn = nn.BatchNorm2d(width)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.spectral(x) + self.skip(x)))


class FFNO(BaseSurrogate):
    """Factorized Fourier Neural Operator (F-FNO).

    Args:
        n_c: channel width of the lifted representation (default 32).
        n_layers: number of stacked F-FNO blocks (default 4).
        modes: Fourier modes per spatial dimension (default 32).
        activation: activation function, 'gelu' or 'relu' (default 'gelu').
        lifting_factor: input lifting multiplier (default 1).
        training: dict of training extras (scheduler, wd, etc.) — ignored by model.
    """

    def __init__(
        self,
        n_c: int = 32,
        n_layers: int = 4,
        modes: int = 32,
        activation: str = "gelu",
        lifting_factor: int = 1,
        training: dict | None = None,
    ) -> None:
        super().__init__()
        # Ignore training dict (passed by train.py but not used by model)
        modes1 = modes
        modes2 = modes // 2 + 1  # rfft along width gives W//2+1 modes
        modes2 = min(modes2, modes)  # cap at modes

        self.lift = nn.Sequential(
            nn.Conv2d(1, n_c, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            FFNOBlock(n_c, modes1, modes2, activation)
            for _ in range(n_layers)
        ])
        self.project = nn.Sequential(
            nn.Conv2d(n_c, n_c, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(n_c, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for blk in self.blocks:
            h = blk(h)
        return self.project(h)


if __name__ == "__main__":
    # Quick smoke test
    for n_c in [16, 32, 64]:
        for modes in [16, 32]:
            m = FFNO(n_c=n_c, modes=modes, n_layers=4)
            n_params = sum(p.numel() for p in m.parameters())
            x = torch.randn(2, 1, 640, 640)
            with torch.no_grad():
                y = m(x)
            print(f"FFNO n_c={n_c} modes={modes}: params={n_params:,} ({n_params/1e6:.1f}M) out={tuple(y.shape)} min={y.min():.4f}")


class Model(FFNO):
    """Training entrypoint for generated Auto V5 runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)
