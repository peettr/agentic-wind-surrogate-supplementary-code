import torch
import torch.nn as nn
import torch.nn.functional as F


class _ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.conv1 = _ReflectionConv2d(channels, channels, 3)
        self.conv2 = _ReflectionConv2d(channels, channels, 3)
        self.act = nn.GELU()
        self.res_scale = res_scale

    def forward(self, x):
        r = self.conv2(self.act(self.conv1(x)))
        return x + r * self.res_scale


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = _ReflectionConv2d(in_channels, out_channels, 3)
        self.res1 = _ResidualBlock(out_channels)
        self.res2 = _ResidualBlock(out_channels)

    def forward(self, x):
        x = self.proj(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = _ReflectionConv2d(in_channels + skip_channels, out_channels, 3)
        self.res1 = _ResidualBlock(out_channels)
        self.res2 = _ResidualBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class edsr_residual_head_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.head = _ReflectionConv2d(in_channels, channels[0], 3)

        self.encoders = nn.ModuleList([
            _EncoderBlock(ch, ch)
            for ch in channels
        ])

        self.downs = nn.ModuleList([
            _ReflectionConv2d(channels[i], channels[i + 1], 3, stride=2)
            for i in range(depth - 1)
        ])

        self.bottleneck = nn.Sequential(
            _ResidualBlock(channels[-1]),
            _ResidualBlock(channels[-1]),
            _ResidualBlock(channels[-1]),
            _ResidualBlock(channels[-1]),
        )

        self.decoders = nn.ModuleList([
            _DecoderBlock(channels[i + 1], channels[i], channels[i])
            for i in range(depth - 2, -1, -1)
        ])

        self.tail = nn.Sequential(
            _ReflectionConv2d(channels[0], channels[0], 3),
            nn.GELU(),
            _ReflectionConv2d(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        h = self.head(x_masked)
        skips = []

        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            h = dec(h, skip)

        y = self.tail(h)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid = valid.expand(-1, y.shape[1], -1, -1)
        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y


