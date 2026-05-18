import torch
import torch.nn as nn
import torch.nn.functional as F

class lowrank_kernel_unet(nn.Module):
    class RefConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                lowrank_kernel_unet.RefConv2d(in_channels, out_channels, 3, 1, False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                lowrank_kernel_unet.RefConv2d(out_channels, out_channels, 3, 1, False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else lowrank_kernel_unet.RefConv2d(in_channels, out_channels, 1, 1, False)
            )

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class LowRankKernelBlock(nn.Module):
        def __init__(self, channels, rank=16):
            super().__init__()
            rank = min(rank, channels)
            self.local = lowrank_kernel_unet.ConvBlock(channels, channels)
            self.reduce = nn.Conv2d(channels, rank, 1, bias=False)
            self.expand = nn.Conv2d(rank, channels, 1, bias=False)
            self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x):
            y = self.local(x)
            context = F.adaptive_avg_pool2d(self.reduce(y), 1)
            context = self.expand(context)
            return y + self.gate * context

    def __init__(self, in_channels=1, out_channels=1, n_c=32, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self.ConvBlock(in_channels, channels[0])

        self.downs = nn.ModuleList()
        self.encoders = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(self.RefConv2d(channels[i - 1], channels[i], 3, 2, False))
            self.encoders.append(self.ConvBlock(channels[i], channels[i]))

        self.bottleneck = self.LowRankKernelBlock(channels[-1], rank=max(8, n_c // 2))

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1, bias=False))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.RefConv2d(channels[0], channels[0], 3, 1, False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            self.RefConv2d(channels[0], out_channels, 1, 1, True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for down, encoder in zip(self.downs, self.encoders):
            y = down(y)
            y = encoder(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = torch.cat([y, skip], dim=1)
            y = decoder(y)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != y.shape[1]:
            out_valid = out_valid[:, :1].expand(-1, y.shape[1], -1, -1)

        return torch.where(out_valid, y, torch.full_like(y, float("nan")))


