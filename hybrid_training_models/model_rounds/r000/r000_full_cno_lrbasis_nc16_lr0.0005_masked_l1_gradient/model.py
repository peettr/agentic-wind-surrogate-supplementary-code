import torch
import torch.nn as nn
import torch.nn.functional as F

class cno_lrbasis(nn.Module):
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

        def forward(self, x):
            return self.net(x)

    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, ch), ch),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, ch), ch),
            )

        def forward(self, x):
            return F.gelu(x + self.net(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.ConvBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(self.ConvBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            self.ResBlock(channels[-1]),
            self.ResBlock(channels[-1]),
            self.ResBlock(channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self.ConvBlock(channels[i + 1] + channels[i], channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for block in self.encoder:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            h = block(h)
            skips.append(h)

        h = self.bottleneck(h)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != self.out_channels:
            valid = valid.any(dim=1, keepdim=True).expand(-1, self.out_channels, -1, -1)

        out = torch.where(valid, out, torch.full_like(out, float("nan")))
        return out