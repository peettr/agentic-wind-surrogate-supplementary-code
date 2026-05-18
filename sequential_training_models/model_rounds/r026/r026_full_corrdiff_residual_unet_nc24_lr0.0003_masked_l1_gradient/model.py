import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv2d(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _ReflectConv2d(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.silu(y + self.skip(x))


class corrdiff_residual_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(_ResidualBlock(prev_channels, ch))
            prev_channels = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            _ResidualBlock(channels[-1], channels[-1]),
            _ResidualBlock(channels[-1], channels[-1]),
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(channels[i + 1], channels[i], 1))
            self.decoders.append(_ResidualBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            _ReflectConv2d(channels[0], channels[0], 3),
            nn.SiLU(),
            _ReflectConv2d(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for i, enc in enumerate(self.encoders):
            y = enc(y)
            skips.append(y)
            if i < len(self.encoders) - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for up_conv, dec, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_conv(y)
            y = torch.cat([y, skip], dim=1)
            y = dec(y)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if y.shape[1] != valid.shape[1]:
            valid = valid[:, :1].expand(-1, y.shape[1], -1, -1)

        return torch.where(valid, y, torch.full_like(y, float("nan")))