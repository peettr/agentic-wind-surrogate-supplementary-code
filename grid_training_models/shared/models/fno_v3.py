"""2D Fourier Neural Operator conforming to the BaseSurrogate contract.

Standard FNO (Li et al., 2020) adapted to the locked ``(B, 1, 640, 640)``
contract. Includes a learnable lift, ``depth`` spectral blocks with pointwise
skip connections, and a ReLU output head (Appendix A.4, non-negative winds).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSurrogate


class SpectralConv2d(nn.Module):
    """2D spectral convolution (truncated FFT + complex multiply)."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        # Two learnable weight tensors for the low-frequency corners in rFFT space.
        self.w1 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )
        self.w2 = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat)
        )

    @staticmethod
    def _complex_mul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (B, I, m1, m2) × (I, O, m1, m2) → (B, O, m1, m2)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            b, self.out_ch, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self._complex_mul(
            x_ft[:, :, : self.modes1, : self.modes2], self.w1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._complex_mul(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.w2
        )
        return torch.fft.irfft2(out_ft, s=x.shape[-2:], norm="ortho")


class FNOBlock(nn.Module):
    """Spectral conv + pointwise conv skip + GELU."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes, modes)
        self.skip = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.skip(x))


class FNO2d(BaseSurrogate):
    """Fourier Neural Operator with ``depth`` spectral blocks.

    Args:
        modes: Fourier modes retained per spatial dimension (default 12).
        width: channel width of the lifted representation (default 32).
        depth: number of stacked FNO blocks (default 4).
    """

    def __init__(self, modes: int = 12, width: int = 32, depth: int = 4) -> None:
        super().__init__()
        self.lift = nn.Conv2d(1, width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock(width, modes) for _ in range(depth))
        self.project = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for blk in self.blocks:
            h = blk(h)
        return self.project(h)


if __name__ == "__main__":
    m = FNO2d(modes=12, width=32, depth=4)
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.randn(1, 1, 640, 640)
    with torch.no_grad():
        y = m(x)
    print(f"FNO2d params={n_params:,}  out={tuple(y.shape)}")
