import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """Parameterized encoder–decoder UNet.

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
            # After _pad_cat we concatenate the skip (same channel count) → 2*ch.
            self.dec_blocks.append(ConvBlock(2 * ch, ch))
            prev = ch

        # Output head — v2 exact: 1x1 conv followed by ReLU (non-negative wind speed).
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

class ConvBlock(nn.Module):
    """Two Conv3x3 + BN + ReLU layers using reflection padding."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, padding=0, bias=False),
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
        pad = [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2]
        x = F.pad(x, pad, mode="reflect")
    return torch.cat([x, skip], dim=1)


class unet_v3(nn.Module):
    """Parameterized encoder-decoder UNet v3."""

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_c: int = 16,
        depth: int = 7,
    ) -> None:
        super().__init__()
        if depth not in self.SUPPORTED_DEPTHS:
            raise ValueError(f"depth must be in {self.SUPPORTED_DEPTHS}, got {depth}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth
        self.pool = nn.MaxPool2d(2, 2)

        enc_channels = [n_c * (2 ** k) for k in range(depth)]
        bottleneck_ch = enc_channels[-1] * 2

        self.enc_blocks = nn.ModuleList()
        prev = in_channels
        for ch in enc_channels:
            self.enc_blocks.append(ConvBlock(prev, ch))
            prev = ch

        self.bottleneck = ConvBlock(enc_channels[-1], bottleneck_ch)

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
            nn.Conv2d(n_c, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        finite_mask = torch.isfinite(x)
        x = torch.where(finite_mask, x, torch.zeros_like(x))

        skips: list[torch.Tensor] = []
        h = x
        for i, enc in enumerate(self.enc_blocks):
            h = enc(h if i == 0 else self.pool(h))
            skips.append(h)

        h = self.bottleneck(self.pool(skips[-1]))

        for up, dec, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            h = dec(_pad_cat(up(h), skip))

        return self.out_conv(h)


UNet = unet_v3


if __name__ == "__main__":
    for d in unet_v3.SUPPORTED_DEPTHS:
        m = unet_v3(depth=d, n_c=16)
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.randn(1, 1, 640, 640)
        with torch.no_grad():
            y = m(x)
        print(f"depth={d}: params={n_params:,}  out={tuple(y.shape)}")