"""ConvNeXt V2 UNet conforming to BaseSurrogate contract.

Uses ConvNeXt V2 blocks (depthwise conv + LayerNorm + GELU + GroupNorm) as drop-in
replacement for standard ConvBlock in UNet encoder-decoder. Modernized architecture
with inverted bottleneck design and larger kernel sizes (7x7).

Reference: Woo et al., 2023 "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"

Input/output contract: (B, 1, 640, 640) -> (B, 1, 640, 640) with ReLU output.
Uses GroupNorm (no BatchNorm) for EMA compatibility.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSurrogate


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block: DWConv -> LN -> PWConv (expand) -> GELU -> PWConv (contract) -> GRN.

    Inverted bottleneck with 7x7 depthwise convolution and GroupNorm.
    """

    def __init__(self, dim: int, layer_scale_init: float = 1e-6, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.GroupNorm(min(32, dim), dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        # GRN (Global Response Normalization) from ConvNeXt V2
        self.grn_beta = nn.Parameter(torch.zeros(dim))
        self.grn_gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # GroupNorm needs (B,C,H,W)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # GRN on output (B, H, W, C)
        gx = torch.norm(x, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        x = self.grn_gamma * x * nx + self.grn_beta + x
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # back to (B, C, H, W)
        return residual + x


class ConvNeXtV2Down(nn.Module):
    """Downsample: ConvNeXtV2Block + strided depthwise conv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvNeXtV2Block(in_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.block(x))


class ConvNeXtV2Up(nn.Module):
    """Upsample: transposed conv + ConvNeXtV2Block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvNeXtV2Block(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.up(x))


class ConvNeXtV2UNet(BaseSurrogate):
    """UNet with ConvNeXt V2 blocks replacing standard ConvBlocks.

    Modern inverted-bottleneck design with 7x7 depthwise convolutions,
    LayerNorm-style normalization, and GRN (Global Response Normalization).
    All GroupNorm for EMA compatibility.

    Args:
        n_c: base channel width (default 16).
        depth: number of encoder/decoder stages (default 5).
        kernel_size: depthwise conv kernel size (default 7).
        training: dict of training extras â€” ignored by model.
    """

    def __init__(
        self,
        n_c: int = 16,
        depth: int = 5,
        kernel_size: int = 7,
        training: dict | None = None,
    ) -> None:
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, n_c, kernel_size=4, stride=4, padding=0),
            nn.GroupNorm(min(32, n_c), n_c),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        channels = [n_c]
        for i in range(depth):
            in_ch = channels[-1]
            out_ch = in_ch * 2
            self.encoders.append(ConvNeXtV2Down(in_ch, out_ch))
            channels.append(out_ch)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvNeXtV2Block(channels[-1]),
            ConvNeXtV2Block(channels[-1]),
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth):
            in_ch = channels[-1]
            out_ch = in_ch // 2
            self.decoders.append(ConvNeXtV2Up(in_ch + out_ch, out_ch))  # +out_ch for skip
            channels.append(out_ch)

        # Head: upsample back to 640x640
        self.head = nn.Sequential(
            nn.ConvTranspose2d(channels[-1], n_c, kernel_size=4, stride=4, padding=0),
            nn.Conv2d(n_c, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.channels = channels
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem: 640 -> 160
        h = self.stem(x)

        # Encoder
        skips = [h]
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            # Upsample h to match skip size
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        # Head: 160 -> 640
        out = self.head(h)
        # Ensure output matches input size
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


if __name__ == "__main__":
    for n_c in [16, 32]:
        for depth in [4, 5]:
            m = ConvNeXtV2UNet(n_c=n_c, depth=depth)
            n_params = sum(p.numel() for p in m.parameters())
            x = torch.randn(2, 1, 640, 640)
            with torch.no_grad():
                y = m(x)
            print(f"ConvNeXtV2UNet n_c={n_c} depth={depth}: params={n_params:,} ({n_params/1e6:.1f}M) out={tuple(y.shape)} min={y.min():.4f}")



