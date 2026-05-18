import torch
import torch.nn as nn
import torch.nn.functional as F

class RefConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            RefConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            RefConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            RefConv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.net(x) + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = RefConv2d(in_channels, out_channels, 1, bias=False)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class evolution_detail_decomp_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [n_c * min(2 ** i, 8) for i in range(depth)]

        self.stem = ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            DownBlock(channels[i - 1], channels[i]) for i in range(1, depth)
        )
        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList(
            UpBlock(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        )

        self.head = nn.Sequential(
            RefConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            RefConv2d(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for down in self.encoder:
            h = down(h)
            skips.append(h)

        h = self.bottleneck(h)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            h = up(h, skip)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != out.shape[1]:
            valid_out = valid_out[:, :1].expand(-1, out.shape[1], -1, -1)

        return torch.where(valid_out, out, torch.full_like(out, float("nan")))


