"""Generated standalone Grid model for unet_v3.

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



class ConvBlock(nn.Module):
    """Two Conv3x3 + BN + ReLU layers. Byte-identical to v2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Pad ``x`` to ``skip``'s spatial size, then concatenate on channel axis."""
    dh = skip.size(2) - x.size(2)
    dw = skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
    return torch.cat([x, skip], dim=1)


class UNet(BaseSurrogate):
    """Parameterized encoderâ€“decoder UNet.

    Args:
        depth: number of encoder stages (5, 6 or 7). Each stage halves resolution.
        n_c:   base channel count; channel widths double per stage (default 16).
    """

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, depth: int = 7, n_c: int = 16) -> None:
        super().__init__()
        if depth not in self.SUPPORTED_DEPTHS:
            raise ValueError(f"depth must be in {self.SUPPORTED_DEPTHS}, got {depth}")
        self.depth = depth
        self.n_c = n_c
        self.pool = nn.MaxPool2d(2, 2)

        enc_channels = [n_c * (2 ** k) for k in range(depth)]
        bottleneck_ch = enc_channels[-1] * 2

        self.enc_blocks = nn.ModuleList()
        prev = 1  # single-channel input
        for ch in enc_channels:
            self.enc_blocks.append(ConvBlock(prev, ch))
            prev = ch
        self.bottleneck = ConvBlock(enc_channels[-1], bottleneck_ch)

        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        prev = bottleneck_ch
        for ch in reversed(enc_channels):
            # v2-identical ConvTranspose2d: kernel=3, stride=2, padding=1, output_padding=1.
            self.up_blocks.append(
                nn.ConvTranspose2d(prev, ch, 3, stride=2, padding=1, output_padding=1)
            )
            # After _pad_cat we concatenate the skip (same channel count) â†’ 2*ch.
            self.dec_blocks.append(ConvBlock(2 * ch, ch))
            prev = ch

        # Output head â€” v2 exact: 1x1 conv followed by ReLU (non-negative wind speed).
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
            h = dec(_pad_cat(up(h), skip))
        return self.out_conv(h)


if __name__ == "__main__":
    for d in UNet.SUPPORTED_DEPTHS:
        m = UNet(depth=d, n_c=16)
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.randn(1, 1, 640, 640)
        with torch.no_grad():
            y = m(x)
        print(f"depth={d}: params={n_params:,}  out={tuple(y.shape)}")


class Model(UNet):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



