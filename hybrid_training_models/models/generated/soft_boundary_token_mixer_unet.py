import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels):
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class RefConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class SoftBoundaryTokenMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = RefConv2d(channels, channels, kernel_size=3, groups=channels, bias=True)
        self.wide = RefConv2d(channels, channels, kernel_size=5, groups=channels, bias=True)
        self.gate = nn.Sequential(
            RefConv2d(channels, channels, kernel_size=3, groups=channels, bias=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mix = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        g = self.gate(x)
        y = g * self.local(x) + (1.0 - g) * self.wide(x)
        return self.mix(y)


class ConvMixerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = RefConv2d(in_channels, out_channels, kernel_size=3)
        self.norm1 = _gn(out_channels)
        self.mixer = SoftBoundaryTokenMixer(out_channels)
        self.norm2 = _gn(out_channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
        )
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.norm1(self.proj(x)))
        x = x + self.mixer(self.norm2(x))
        x = x + self.ffn(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = RefConv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.block = ConvMixerBlock(out_channels, out_channels)

    def forward(self, x):
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = RefConv2d(in_channels + skip_channels, out_channels, kernel_size=3)
        self.block = ConvMixerBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(self.reduce(x))


class soft_boundary_token_mixer_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = ConvMixerBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            DownBlock(channels[i - 1], channels[i]) for i in range(1, depth)
        )

        self.bottleneck = nn.Sequential(
            ConvMixerBlock(channels[-1], channels[-1]),
            ConvMixerBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList(
            UpBlock(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        )

        self.head = nn.Sequential(
            RefConv2d(channels[0], channels[0], kernel_size=3),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for block in self.encoder:
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = block(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)

        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y