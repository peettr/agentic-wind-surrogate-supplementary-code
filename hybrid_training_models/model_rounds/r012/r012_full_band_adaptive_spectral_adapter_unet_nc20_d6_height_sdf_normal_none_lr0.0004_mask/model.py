import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels):
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ReflectConv2d(in_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            ReflectConv2d(out_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SpectralAdapter(nn.Module):
    def __init__(self, channels, modes=24):
        super().__init__()
        self.modes = modes
        self.real_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.imag_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            _gn(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)

        x_ft = torch.fft.rfft2(x, norm="ortho")
        low = x_ft[:, :, :mh, :mw]
        scale = torch.complex(self.real_scale, self.imag_scale)
        low = low * scale

        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :mh, :mw] = low
        y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return x + self.mix(y)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(F.avg_pool2d(x, kernel_size=2, stride=2))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class band_adaptive_spectral_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.input_block = ConvBlock(in_channels, channels[0])
        self.down_blocks = nn.ModuleList(
            DownBlock(channels[i - 1], channels[i]) for i in range(1, depth)
        )

        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_channels, bottleneck_channels),
            SpectralAdapter(bottleneck_channels),
            ConvBlock(bottleneck_channels, bottleneck_channels),
        )

        self.up_blocks = nn.ModuleList(
            UpBlock(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        )

        self.output_head = nn.Sequential(
            ReflectConv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.input_block(x_masked)
        skips.append(h)

        for down in self.down_blocks:
            h = down(h)
            skips.append(h)

        h = self.bottleneck(h)

        for up, skip in zip(self.up_blocks, reversed(skips[:-1])):
            h = up(h, skip)

        out = self.output_head(h)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] == out.shape[1]:
            out_valid = valid
        else:
            out_valid = valid.all(dim=1, keepdim=True).expand_as(out)

        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out