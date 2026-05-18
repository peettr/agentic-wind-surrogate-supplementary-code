"""AFNO â€” Adaptive Fourier Neural Operator for 2D dense regression.

Improves on FNO by using adaptive spectral mixing with channel-wise attention
and block-diagonal weight matrices in frequency domain. Better scalability
to high-resolution inputs.

Based on: Guibas et al., 2022 (Adaptive Fourier Neural Operators)
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


class AFNOLayer(nn.Module):
    """Adaptive Fourier layer with block-diagonal spectral mixing and mode limiting."""

    def __init__(self, hidden_size: int, num_blocks: int = 8, sparsity_threshold: float = 0.01,
                 hard_thresholding_fraction: float = 1.0, max_modes: int = 0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.max_modes = max_modes  # 0 = use hard_thresholding_fraction (old behavior)

        # Block-diagonal weights for spectral mixing (two-layer MLP in frequency domain)
        self.w1 = nn.Parameter(0.02 * torch.randn(num_blocks, self.block_size, self.block_size))
        self.w2 = nn.Parameter(0.02 * torch.randn(num_blocks, self.block_size, self.block_size))
        self.b1 = nn.Parameter(torch.zeros(num_blocks, self.block_size))
        self.b2 = nn.Parameter(torch.zeros(num_blocks, self.block_size))

        # Channel mixing MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden_size * 2, hidden_size, 1),
        )
        self.norm = _gn(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)

        # FFT
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # Mode selection: use max_modes if set, otherwise hard_thresholding_fraction
        if self.max_modes > 0:
            h_modes = min(H, self.max_modes)
            w_modes = min(x_ft.shape[-1], self.max_modes)
        else:
            h_modes = max(1, int(H * self.hard_thresholding_fraction))
            w_modes = max(1, int(x_ft.shape[-1] * self.hard_thresholding_fraction))

        # Reshape for block-diagonal multiplication
        x_ft = x_ft[:, :, :h_modes, :w_modes]  # (B, C, h_m, w_m)
        x_ft = x_ft.permute(0, 3, 2, 1).reshape(B, w_modes, h_modes, self.num_blocks, self.block_size)

        # Two-layer spectral mixing: w1 â†’ GELU â†’ w2 (FIXED: was only using w1)
        x_r = x_ft.real
        x_i = x_ft.imag

        # Layer 1: w1
        x_r = torch.einsum("bwhkc,kcd->bwhkd", x_r, self.w1) + self.b1
        x_i = torch.einsum("bwhkc,kcd->bwhkd", x_i, self.w1) + self.b1

        # Nonlinearity on magnitude (soft gating)
        x_r = F.gelu(x_r)
        x_i = F.gelu(x_i)

        # Layer 2: w2
        x_r = torch.einsum("bwhkc,kcd->bwhkd", x_r, self.w2) + self.b2
        x_i = torch.einsum("bwhkc,kcd->bwhkd", x_i, self.w2) + self.b2

        x_ft = torch.complex(x_r, x_i)

        # Reshape back
        x_ft = x_ft.reshape(B, w_modes, h_modes, C)
        x_ft = x_ft.permute(0, 3, 2, 1)  # (B, C, h_m, w_m)

        # Soft thresholding
        x_ft = x_ft * (torch.abs(x_ft) > self.sparsity_threshold)

        # iFFT
        out = torch.fft.irfft2(x_ft, s=(H, W), norm="ortho")

        # MLP + residual
        out = out + self.mlp(residual)
        return out + residual


class AFNO(BaseSurrogate):
    """Adaptive Fourier Neural Operator for dense wind field prediction.

    U-shaped encoder-decoder with AFNO spectral layers.

    Args:
        width: base channel width.
        num_blocks: number of diagonal blocks in spectral mixing.
        depth: U-Net depth (encoder stages).
        n_layers: AFNO layers per stage.
        max_modes: maximum number of frequency modes to process (0 = all).
    """

    def __init__(self, width: int = 48, num_blocks: int = 8, depth: int = 4,
                 n_layers: int = 2, max_modes: int = 0) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1, bias=False),
            _gn(width), nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = width
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(*[AFNOLayer(ch, num_blocks, max_modes=max_modes) for _ in range(n_layers)]))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            ch *= 2

        self.bottleneck = nn.Sequential(*[AFNOLayer(ch, num_blocks, max_modes=max_modes) for _ in range(n_layers)])

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False), _gn(ch // 2), nn.GELU()))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False), _gn(ch // 2), nn.GELU(),
                *[AFNOLayer(ch // 2, num_blocks, max_modes=max_modes) for _ in range(n_layers)],
            ))
            ch //= 2

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
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh != 0 or dw != 0:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[k](x)
        return self.output_proj(x)



