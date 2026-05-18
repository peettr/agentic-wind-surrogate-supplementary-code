import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            _ReflectConv(in_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            _ReflectConv(out_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        return self.block(x) + self.skip(x)


class _LaplaceRationalFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.a0 = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.a1 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.b1 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        xp = self.pad(x)
        c = xp[:, :, 1:-1, 1:-1]
        lap = xp[:, :, :-2, 1:-1] + xp[:, :, 2:, 1:-1] + xp[:, :, 1:-1, :-2] + xp[:, :, 1:-1, 2:] - 4.0 * c
        denom = 1.0 + F.softplus(self.b1).clamp_min(1.0e-6)
        return (self.a0 * x + self.a1 * lap) / denom


class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv = _ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class laplace_rational_filter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(_ConvBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            _ConvBlock(channels[-1], channels[-1]),
            _LaplaceRationalFilter(channels[-1]),
            _ConvBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(_UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            h = decoder(h, skip)

        output = self.head(h)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid[:, :1].expand(-1, output.shape[1], -1, -1)

        output = output.clone()
        output[~valid] = float("nan")
        return output


