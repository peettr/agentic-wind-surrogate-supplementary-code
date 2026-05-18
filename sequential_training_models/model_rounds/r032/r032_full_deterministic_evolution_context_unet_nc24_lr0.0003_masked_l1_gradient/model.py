import torch
import torch.nn as nn
import torch.nn.functional as F

class deterministic_evolution_context_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class _UpBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.reduce = nn.Sequential(
                nn.ReflectionPad2d(0),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )
            self.conv = deterministic_evolution_context_unet._ConvBlock(
                out_channels + skip_channels,
                out_channels,
            )

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        max_channels = n_c * 8
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]
        self.channels = channels

        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoder.append(self._ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._ConvBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self._UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(0),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i != self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = block(h, skip)

        out = self.head(h)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != self.out_channels:
            valid = valid.all(dim=1, keepdim=True).expand(-1, self.out_channels, -1, -1)

        return torch.where(valid, out, torch.full_like(out, float("nan")))