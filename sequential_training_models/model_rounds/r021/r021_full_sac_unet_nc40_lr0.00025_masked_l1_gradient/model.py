import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_nan(x: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(x), torch.zeros((), dtype=x.dtype, device=x.device), x)


def _gn(ch: int) -> nn.GroupNorm:
    g = next(x for x in range(min(32, ch), 0, -1) if ch % x == 0)
    return nn.GroupNorm(g, ch)


class SpatialAdaptiveConv(nn.Module):
    """Conv2d where kernel weights are predicted per-pixel from a context branch.

    Uses a practical grouped approach: standard conv + per-pixel affine modulation
    predicted from a context branch. This gives spatially-varying filter behavior
    without the prohibitive memory cost of full per-pixel kernels at 640x640.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, groups: int = 1) -> None:
        super().__init__()
        self.out_ch = out_ch
        self.ks = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            padding=kernel_size // 2,
            padding_mode="reflect",
            groups=groups,
            bias=False,
        )
        self.ctx = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, padding_mode="reflect", bias=False),
            _gn(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch * 2, 1, bias=False),
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _mask_nan(x)
        out = self.conv(x)
        params = self.ctx(x)
        scale, shift = params.chunk(2, dim=1)
        scale = 1.0 + torch.tanh(scale)
        return out * scale + shift + self.bias.view(1, -1, 1, 1)


class SACConv(nn.Module):
    """Practical Spatial Adaptive Conv: standard conv + spatial attention modulation."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect", bias=False)
        self.gn = _gn(out_ch)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode="reflect", bias=False),
            _gn(out_ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _mask_nan(x)
        out = self.conv(x)
        attn = self.spatial_attn(x)
        return self.gn(out * attn)


class SACConvBlock(nn.Module):
    """Two SACConv + ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            SACConv(in_ch, out_ch),
            nn.ReLU(inplace=True),
            SACConv(out_ch, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    dh = skip.size(2) - x.size(2)
    dw = skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2], mode="reflect")
    return torch.cat([x, skip], dim=1)


class sac_unet(nn.Module):
    """UNet with Spatial Adaptive Convolution blocks."""

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, in_channels: int = 1, out_channels: int = 1, n_c: int = 16, depth: int = 7) -> None:
        super().__init__()
        if depth not in self.SUPPORTED_DEPTHS:
            raise ValueError(f"depth must be in {self.SUPPORTED_DEPTHS}, got {depth}")
        self.depth = depth
        self.n_c = n_c

        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch_in = in_channels
        for k in range(depth):
            ch_out = n_c * 2 ** k
            self.enc.append(SACConvBlock(ch_in, ch_out))
            self.pool.append(nn.MaxPool2d(2))
            ch_in = ch_out

        bottleneck_ch = n_c * 2 ** depth
        self.bottleneck = SACConvBlock(ch_in, bottleneck_ch)

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch_in = bottleneck_ch
        for k in reversed(range(depth)):
            ch_skip = n_c * 2 ** k
            self.up.append(nn.ConvTranspose2d(ch_in, ch_skip, 2, stride=2))
            self.dec.append(SACConvBlock(ch_skip * 2, ch_skip))
            ch_in = ch_skip

        self.head = nn.Sequential(nn.Conv2d(n_c, out_channels, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _mask_nan(x)
        skips = []
        for enc_block, pool in zip(self.enc, self.pool):
            x = enc_block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            x = _pad_cat(x, skip)
            x = self.dec[k](x)
        return self.head(x)