import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ReflectConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            ReflectConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        return self.net(x) + self.skip(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class unet_v3_pointrefine(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        dec_in = channels[-1]
        for skip_ch in reversed(channels):
            self.decoders.append(UpBlock(dec_in, skip_ch, skip_ch))
            dec_in = skip_ch

        self.refine = nn.Sequential(
            ConvBlock(channels[0], channels[0]),
            ReflectConv2d(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoders, reversed(skips)):
            h = dec(h, skip)

        h = self.refine(h)

        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)

        h = torch.where(valid[:, :1], h, torch.full_like(h, float("nan")))
        return h


