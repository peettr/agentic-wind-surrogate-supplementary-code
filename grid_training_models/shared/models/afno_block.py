"""AFNO (Adaptive Fourier Neural Operator) block for UNet bottleneck.

Reference: Guibas et al. 2022 "Adaptive Fourier Neural Operators:
Efficient and Stable Transformers for Symbolic Regression"

Uses FFT-based global mixing at the bottleneck resolution (~5Ã—5 to 20Ã—20),
making it extremely cheap while providing global receptive field.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFNOBlock(nn.Module):
    """Single AFNO spectral mixing layer.

    At bottleneck resolution (HÃ—W small), FFT is cheap and provides
    global receptive field over the entire spatial domain.
    """

    def __init__(self, hidden_dim: int, num_modes: int = 32, sparsity_threshold: float = 0.01):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.sparsity_threshold = sparsity_threshold

        self.norm = nn.LayerNorm(hidden_dim)
        # MLP in frequency domain: operate on real and imag parts separately
        # Input dim = 2*hidden_dim (stacked real + imag)
        self.mlp_freq = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )
        # Channel mixing MLP (pointwise, real-valued)
        self.mlp_channel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # Learnable soft thresholding
        self.soft_threshold = nn.Parameter(torch.ones(1) * sparsity_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature tensor
        Returns:
            (B, C, H, W) with global spectral mixing applied
        """
        B, C, H, W = x.shape
        residual = x

        # Reshape to (B, H*W, C) for LayerNorm
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # back to (B, C, H, W)

        # 2D FFT
        x_fft = torch.fft.rfft2(x, norm="ortho")
        
        # Truncate to num_modes in each dimension
        modes_h = min(self.num_modes, H)
        modes_w = min(self.num_modes, x_fft.shape[-1])  # rfft halves last dim
        
        # Create mask for low-frequency modes
        mask = torch.zeros(H, x_fft.shape[-1], device=x.device, dtype=torch.bool)
        mask[:modes_h, :modes_w] = True
        # Also keep symmetric high-frequency modes
        if H > modes_h:
            mask[-modes_h:, :modes_w] = True
        
        # Apply spectral mixing only on selected modes
        x_fft_masked = x_fft * mask.unsqueeze(0).unsqueeze(0)
        
        # Split real/imag, apply MLP in frequency domain
        x_real = x_fft_masked.real.permute(0, 2, 3, 1)  # (B, H, W_half, C)
        x_imag = x_fft_masked.imag.permute(0, 2, 3, 1)  # (B, H, W_half, C)
        x_ri = torch.cat([x_real, x_imag], dim=-1)  # (B, H, W_half, 2C)
        orig_shape = x_ri.shape
        x_ri = x_ri.reshape(B, -1, self.hidden_dim * 2)  # (B, H*W_half, 2C)
        x_ri = self.mlp_freq(x_ri)
        
        # Soft thresholding
        x_ri = torch.where(
            x_ri.abs() > self.soft_threshold,
            x_ri,
            torch.zeros_like(x_ri),
        )
        
        x_ri = x_ri.reshape(orig_shape)
        # Reapply mask: zero out non-selected frequency positions (MLP bias leaks)
        mask_spatial = mask.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, self.hidden_dim * 2)  # (1,H,W_half,2C)
        x_ri = x_ri * mask_spatial
        x_real_out = x_ri[..., :self.hidden_dim]
        x_imag_out = x_ri[..., self.hidden_dim:]
        x_fft_out = torch.complex(x_real_out, x_imag_out).permute(0, 3, 1, 2)  # (B, C, H, W_half)
        
        # Inverse FFT
        x_out = torch.fft.irfft2(x_fft_out, s=(H, W), norm="ortho")
        
        # Channel mixing
        x_out = x_out.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_out = self.mlp_channel(x_out)
        x_out = x_out.permute(0, 3, 1, 2)  # back to (B, C, H, W)
        
        return residual + x_out


class AFNOBottleneck(nn.Module):
    """AFNO bottleneck block: optional LayerNorm + N AFNO layers + optional ConvBlock.

    Drop-in replacement for the standard ConvBlock at the UNet bottleneck.
    """

    def __init__(self, in_channels: int, num_afno_layers: int = 1, num_modes: int = 32):
        super().__init__()
        self.afno_layers = nn.ModuleList([
            AFNOBlock(in_channels, num_modes=num_modes)
            for _ in range(num_afno_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.afno_layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Test at various bottleneck resolutions
    for H, W in [(5, 5), (10, 10), (20, 20)]:
        for modes in [16, 32, 64]:
            m = AFNOBottleneck(256, num_afno_layers=1, num_modes=modes)
            n_params = sum(p.numel() for p in m.parameters())
            x = torch.randn(2, 256, H, W)
            with torch.no_grad():
                y = m(x)
            print(f"AFNO {H}x{W} modes={modes}: params={n_params:,} in={tuple(x.shape)} out={tuple(y.shape)}")



