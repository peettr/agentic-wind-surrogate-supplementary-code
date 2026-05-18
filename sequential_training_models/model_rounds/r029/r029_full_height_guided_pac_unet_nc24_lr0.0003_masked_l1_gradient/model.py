import torch
import torch.nn as nn
import torch.nn.functional as F

class _RefConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _RefConv(in_channels, out_channels, 3),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            _RefConv(out_channels, out_channels, 3),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _HeightGuidedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.height_proj = nn.Sequential(
            _RefConv(1, channels, 3),
            nn.SiLU(inplace=True),
            _RefConv(channels, channels, 3),
        )
        self.feature_proj = _RefConv(channels, channels, 3)
        self.out_proj = _RefConv(channels, channels, 3)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, feat, height):
        guide = self.height_proj(height)
        gate = torch.sigmoid(guide)
        guided = self.feature_proj(feat) * gate
        return feat + self.norm(self.out_proj(guided))


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = _ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = _RefConv(in_channels, out_channels, 1)
        self.block = _ConvBlock(out_channels + skip_channels, out_channels)
        self.guide = _HeightGuidedBlock(out_channels)

    def forward(self, x, skip, height):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return self.guide(x, height)


class height_guided_pac_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.stem = _ConvBlock(in_channels, channels[0])
        self.encoders = nn.ModuleList(
            [_Down(channels[i - 1], channels[i]) for i in range(1, depth)]
        )

        self.bottleneck = nn.Sequential(
            _ConvBlock(channels[-1], channels[-1]),
            _HeightGuidedBlock(channels[-1]),
        )

        self.decoders = nn.ModuleList(
            [
                _Up(channels[i], channels[i - 1], channels[i - 1])
                for i in range(depth - 1, 0, -1)
            ]
        )

        self.head = nn.Sequential(
            _RefConv(channels[0], channels[0], 3),
            nn.SiLU(inplace=True),
            _RefConv(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        heights = [x_masked]
        height = x_masked

        for enc in self.encoders:
            height = F.interpolate(height, scale_factor=0.5, mode="bilinear", align_corners=False)
            heights.append(height)
            h = enc(h)
            skips.append(h)

        h = self.bottleneck[0](h)
        h = self.bottleneck[1](h, heights[-1])

        for dec, skip, height in zip(self.decoders, reversed(skips[:-1]), reversed(heights[:-1])):
            h = dec(h, skip, height)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            out_valid = out_valid[:, :1].expand(-1, out.shape[1], -1, -1)

        return torch.where(out_valid, out, torch.full_like(out, float("nan")))