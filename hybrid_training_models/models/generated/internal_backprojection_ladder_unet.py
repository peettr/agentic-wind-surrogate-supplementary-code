import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class internal_backprojection_ladder_unet(nn.Module):
    class ReflectConv(nn.Module):
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
                internal_backprojection_ladder_unet.ReflectConv(in_channels, out_channels, 3, bias=False),
                _gn(out_channels),
                nn.SiLU(inplace=True),
                internal_backprojection_ladder_unet.ReflectConv(out_channels, out_channels, 3, bias=False),
                _gn(out_channels),
                nn.SiLU(inplace=True),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
            )

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class UpBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.reduce = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
            self.fuse = internal_backprojection_ladder_unet.ConvBlock(out_channels + skip_channels, out_channels)
            self.refine = internal_backprojection_ladder_unet.ConvBlock(out_channels, out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = self.fuse(torch.cat([x, skip], dim=1))
            x = self.refine(x)
            return x

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self.ConvBlock(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        for i in range(1, depth):
            self.encoders.append(self.ConvBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self.UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ConvBlock(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x).any(dim=1, keepdim=True)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for encoder in self.encoders:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            h = encoder(h)
            skips.append(h)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            h = decoder(h, skip)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid = valid.expand(-1, out.shape[1], -1, -1)
        out = out.clone()
        out[~valid] = float("nan")
        return out