import torch
import torch.nn as nn
import torch.nn.functional as F

class mean_residual_corrector_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            g1 = 8
            while out_ch % g1 != 0 and g1 > 1:
                g1 //= 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, padding=0, bias=False),
                nn.GroupNorm(g1, out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, padding=0, bias=False),
                nn.GroupNorm(g1, out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._ConvBlock(channels[-1], channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoders.append(self._ConvBlock(channels[i] * 2, channels[i]))

        self.mean_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[-1], channels[-1], kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[-1], out_channels, kernel_size=1),
        )

        self.out_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, padding=0),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i < len(self.encoders) - 1:
                h = self.down(h)

        h = self.bottleneck(h)
        mean = self.mean_head(h)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        residual = self.out_head(h)
        output = residual + mean
        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output