import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ReflectionConv2d(in_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.net(x) + self.skip(x)


class SparseTTAdapterRouter(nn.Module):
    def __init__(self, channels, rank=8, experts=4):
        super().__init__()
        rank = min(rank, channels)
        self.experts = experts
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, experts),
        )
        self.down = nn.ModuleList([
            nn.Conv2d(channels, rank, kernel_size=1, bias=False)
            for _ in range(experts)
        ])
        self.up = nn.ModuleList([
            nn.Conv2d(rank, channels, kernel_size=1, bias=False)
            for _ in range(experts)
        ])
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        weights = torch.softmax(self.router(x), dim=1)
        y = torch.zeros_like(x)
        for i in range(self.experts):
            yi = self.up[i](F.silu(self.down[i](x), inplace=True))
            y = y + yi * weights[:, i].view(-1, 1, 1, 1)
        return x + self.scale * y


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class sparse_tt_adapter_router_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.depth = depth
        channels = [n_c * min(2 ** i, 8) for i in range(depth)]

        self.stem = ConvBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(ConvBlock(channels[i - 1], channels[i]))

        self.adapters = nn.ModuleList([
            SparseTTAdapterRouter(c, rank=max(4, min(c // 4, 16)), experts=4)
            for c in channels
        ])

        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], channels[-1]),
            SparseTTAdapterRouter(channels[-1], rank=max(4, min(channels[-1] // 4, 16)), experts=4),
            ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        y = self.adapters[0](y)
        skips.append(y)

        for i, block in enumerate(self.encoder, start=1):
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
            y = block(y)
            y = self.adapters[i](y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = block(y, skip)

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y


