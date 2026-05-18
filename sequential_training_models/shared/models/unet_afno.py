"""UNet with AFNO spectral bottleneck.

Drop-in replacement for the standard 7-level UNet where the bottleneck
is replaced with AFNO (Adaptive Fourier Neural Operator) blocks for
global spectral mixing at the lowest resolution.

At the bottleneck (5×5 or 10×10), FFT is extremely cheap and provides
a true global receptive field over the entire 640×640 input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSurrogate
from .afno_block import AFNOBottleneck


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
        training: dict of training extras — ignored by model.
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
