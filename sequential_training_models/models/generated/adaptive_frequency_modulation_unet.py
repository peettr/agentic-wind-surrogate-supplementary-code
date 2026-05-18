import torch
import torch.nn as nn
import torch.nn.functional as F

class adaptive_frequency_modulation_unet(nn.Module):
    class ReflectionConv2d(nn.Module):
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
                adaptive_frequency_modulation_unet.ReflectionConv2d(in_channels, out_channels, 3, False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                adaptive_frequency_modulation_unet.ReflectionConv2d(out_channels, out_channels, 3, False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class FrequencyModulation(nn.Module):
        def __init__(self, channels, modes=24):
            super().__init__()
            self.modes = modes
            self.gain = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.mix = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(1, channels // 4), 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(1, channels // 4), channels, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            b, c, h, w = x.shape
            modes_h = min(self.modes, h)
            modes_w = min(self.modes, w // 2 + 1)

            xf = torch.fft.rfft2(x, norm="ortho")
            low = torch.zeros_like(xf)
            low[:, :, :modes_h, :modes_w] = xf[:, :, :modes_h, :modes_w]
            low_spatial = torch.fft.irfft2(low, s=(h, w), norm="ortho")

            gate = self.mix(x)
            return x + gate * self.gain * low_spatial

    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.pool = nn.AvgPool2d(2)
            self.conv = adaptive_frequency_modulation_unet.ConvBlock(in_channels, out_channels)

        def forward(self, x):
            return self.conv(self.pool(x))

    class UpBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.conv = adaptive_frequency_modulation_unet.ConvBlock(in_channels + skip_channels, out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            self.DownBlock(channels[i - 1], channels[i]) for i in range(1, depth)
        )

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.FrequencyModulation(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList(
            self.UpBlock(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        )

        self.head = nn.Sequential(
            self.ReflectionConv2d(channels[0], channels[0], 3, False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(0),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for block in self.encoder:
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = block(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)

        return torch.where(valid, y, torch.full_like(y, float("nan")))