"""Generated standalone Grid model for unet_afno.

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
        """Forward pass for Grid generated training source-of-truth models."""

    def check_shapes(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[1:] != (1, 640, 640):
            raise ValueError(f"Input shape mismatch: expected (B, 1, 640, 640), got {tuple(x.shape)}")
        if y.shape[1:] != (1, 640, 640):
            raise ValueError(f"Output shape mismatch: expected (B, 1, 640, 640), got {tuple(y.shape)}")



# Embedded local dependency copy: afno_block.py
"""AFNO (Adaptive Fourier Neural Operator) block for UNet bottleneck.

Reference: Guibas et al. 2022 "Adaptive Fourier Neural Operators:
Efficient and Stable Transformers for Symbolic Regression"

Uses FFT-based global mixing at the bottleneck resolution (~5Ã—5 to 20Ã—20),
making it extremely cheap while providing global receptive field.
"""

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



class ConvBlock(nn.Module):
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

    def forward(self, x):
        return self.block(x)


class UNetAFNO(BaseSurrogate):
    """7-level UNet with AFNO spectral bottleneck.

    Args:
        n_c: base channel width.
        in_channels: number of input channels (1=height, 2=+sdf, 3=+sdf+normal).
        afno_layers: number of AFNO layers in bottleneck (0 = standard UNet).
        afno_modes: number of FFT modes to keep in AFNO.
        training: dict of training extras â€” ignored by model.
    """

    def __init__(
        self,
        n_c: int = 16,
        in_channels: int = 1,
        afno_layers: int = 1,
        afno_modes: int = 32,
        depth: int = 7,
        training: dict | None = None,
    ) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        enc_channels = [n_c * (2 ** k) for k in range(depth)]
        bottleneck_ch = enc_channels[-1] * 2

        self.enc_blocks = nn.ModuleList()
        prev = in_channels
        for ch in enc_channels:
            self.enc_blocks.append(ConvBlock(prev, ch))
            prev = ch

        # Bottleneck: AFNO replaces standard ConvBlock
        if afno_layers > 0:
            self.bottleneck = nn.Sequential(
                ConvBlock(enc_channels[-1], bottleneck_ch),
                AFNOBottleneck(bottleneck_ch, num_afno_layers=afno_layers, num_modes=afno_modes),
            )
        else:
            self.bottleneck = ConvBlock(enc_channels[-1], bottleneck_ch)

        # Decoder (same as standard UNet)
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        prev = bottleneck_ch
        for ch in reversed(enc_channels):
            self.up_blocks.append(
                nn.ConvTranspose2d(prev, ch, 3, stride=2, padding=1, output_padding=1)
            )
            self.dec_blocks.append(ConvBlock(2 * ch, ch))
            prev = ch

        self.out_conv = nn.Sequential(
            nn.Conv2d(n_c, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x
        for i, enc in enumerate(self.enc_blocks):
            h = enc(h if i == 0 else self.pool(h))
            skips.append(h)
        h = self.bottleneck(self.pool(skips[-1]))

        for up, dec, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            h = self._pad_cat(up(h), skip)
            h = dec(h)

        return self.out_conv(h)

    @staticmethod
    def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return torch.cat([x, skip], dim=1)


if __name__ == "__main__":
    for n_c in [16, 32]:
        for afno_layers in [0, 1, 2]:
            for afno_modes in [16, 32]:
                m = UNetAFNO(n_c=n_c, afno_layers=afno_layers, afno_modes=afno_modes, depth=7)
                n_params = sum(p.numel() for p in m.parameters())
                x = torch.randn(2, 1, 640, 640)
                with torch.no_grad():
                    y = m(x)
                print(f"n_c={n_c} afno={afno_layers}x{afno_modes}: params={n_params:,} ({n_params/1e6:.1f}M) out={tuple(y.shape)} min={y.min():.4f}")


class Model(UNetAFNO):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



