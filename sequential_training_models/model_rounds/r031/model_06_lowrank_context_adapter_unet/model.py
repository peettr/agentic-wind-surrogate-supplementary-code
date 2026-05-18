import torch
import torch.nn as nn
import torch.nn.functional as F

class lowrank_context_adapter_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            groups = min(8, out_channels)
            while out_channels % groups != 0:
                groups -= 1

            self.net = nn.Sequential(
                lowrank_context_adapter_unet.ReflectConv(in_channels, out_channels, 3),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
                lowrank_context_adapter_unet.ReflectConv(out_channels, out_channels, 3),
                nn.GroupNorm(groups, out_channels),
            )
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))

    class LowRankContextAdapter(nn.Module):
        def __init__(self, channels, rank=None):
            super().__init__()
            rank = rank or max(8, channels // 8)
            self.down = nn.Conv2d(channels, rank, 1)
            self.up = nn.Conv2d(rank, channels * 2, 1)

        def forward(self, x):
            context = F.adaptive_avg_pool2d(x, 1)
            scale_shift = self.up(F.silu(self.down(context), inplace=True))
            scale, shift = torch.chunk(scale_shift, 2, dim=1)
            return x * (1.0 + torch.tanh(scale)) + shift

    class Down(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = lowrank_context_adapter_unet.ConvBlock(in_channels, out_channels)

        def forward(self, x):
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            return self.block(x)

    class Up(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.block = lowrank_context_adapter_unet.ConvBlock(in_channels + skip_channels, out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return self.block(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.stem = self.ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            [self.Down(channels[i - 1], channels[i]) for i in range(1, depth)]
        )

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.LowRankContextAdapter(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self.Up(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
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

        output = self.head(h)
        output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if output.shape[1] == valid.shape[1]:
            output = torch.where(valid, output, torch.full_like(output, float("nan")))
        else:
            valid_out = valid.any(dim=1, keepdim=True).expand_as(output)
            output = torch.where(valid_out, output, torch.full_like(output, float("nan")))

        return output


