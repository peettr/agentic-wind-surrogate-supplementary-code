import torch
import torch.nn as nn
import torch.nn.functional as F

class geometry_bridge_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(self._conv_block(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        bottleneck_ch = channels[-1]
        self.bottleneck = self._conv_block(bottleneck_ch, bottleneck_ch)

        self.decoder = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            skip_ch = channels[i]
            in_ch = bottleneck_ch + skip_ch
            out_ch = skip_ch
            self.decoder.append(self._conv_block(in_ch, out_ch))
            bottleneck_ch = out_ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0, bias=True),
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(self._groups(out_ch), out_ch),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(self._groups(out_ch), out_ch),
            nn.GELU(),
        )

    def _groups(self, channels):
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        h = x_masked
        skips = []

        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        h = self.head(h)

        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)

        nan_fill = torch.full_like(h, float("nan"))
        return torch.where(valid, h, nan_fill)
