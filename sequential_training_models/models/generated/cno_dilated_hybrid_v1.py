import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = ReflectConv2d(channels, channels, 3, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = ReflectConv2d(channels, channels, 3, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = ReflectConv2d(in_channels, out_channels, 3, stride=2)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.res = ResidualBlock(out_channels)

    def forward(self, x):
        x = F.gelu(self.norm(self.down(x)))
        return self.res(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ReflectConv2d(in_channels + skip_channels, out_channels, 3)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.res = ResidualBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.gelu(self.norm(self.conv(x)))
        return self.res(x)


class cno_dilated_hybrid_v1(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=32, depth=5):
        super().__init__()
        depth = max(1, int(depth))
        channels = [n_c * min(2 ** i, 8) for i in range(depth)]

        self.stem = nn.Sequential(
            ReflectConv2d(in_channels, channels[0], 3),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.GELU(),
            ResidualBlock(channels[0]),
        )

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(DownBlock(channels[i - 1], channels[i]))

        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            ResidualBlock(bottleneck_channels, dilation=1),
            ResidualBlock(bottleneck_channels, dilation=2),
            ResidualBlock(bottleneck_channels, dilation=4),
            ResidualBlock(bottleneck_channels, dilation=8),
            ResidualBlock(bottleneck_channels, dilation=1),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.decoder.append(UpBlock(channels[i], channels[i - 1], channels[i - 1]))

        self.head = nn.Sequential(
            ReflectConv2d(channels[0], channels[0], 3),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.GELU(),
            ReflectConv2d(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for block in self.encoder:
            h = block(h)
            skips.append(h)

        h = self.bottleneck(h)

        skips = skips[:-1][::-1]
        for block, skip in zip(self.decoder, skips):
            h = block(h, skip)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out.shape[1] != out_valid.shape[1]:
            out_valid = out_valid[:, :1].expand(-1, out.shape[1], -1, -1)

        return torch.where(out_valid, out, torch.full_like(out, float("nan")))


