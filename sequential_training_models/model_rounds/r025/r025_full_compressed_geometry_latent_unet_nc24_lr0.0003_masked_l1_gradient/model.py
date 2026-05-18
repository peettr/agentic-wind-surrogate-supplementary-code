import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv2d(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _ReflectConv2d(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.silu(y + self.skip(x))


class compressed_geometry_latent_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoder.append(_Block(prev, ch))
            prev = ch

        self.bottleneck = _Block(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(_Block(channels[i + 1] + channels[i], channels[i]))

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked
        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i < self.depth - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))

        y = self.out_conv(self.out_pad(y))

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return torch.where(valid, y, torch.full_like(y, float("nan")))