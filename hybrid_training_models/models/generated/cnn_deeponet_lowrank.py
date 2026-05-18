import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn_deeponet_lowrank(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class Bottleneck(nn.Module):
        def __init__(self, channels, rank=32):
            super().__init__()
            rank = min(rank, channels)
            self.local = cnn_deeponet_lowrank.ConvBlock(channels, channels)
            self.branch = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, rank, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(rank, channels, kernel_size=1),
                nn.Sigmoid(),
            )
            self.trunk = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3, padding=0, groups=channels, bias=False),
                nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False),
                nn.GELU(),
            )

        def forward(self, x):
            z = self.local(x)
            return z + self.trunk(z) * self.branch(z)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=5):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = self.Bottleneck(channels[-1], rank=max(n_c, channels[-1] // 4))

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1, padding=0))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.head(h)
        out_valid = valid if valid.shape[1] == out.shape[1] else valid.all(dim=1, keepdim=True).expand_as(out)
        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out