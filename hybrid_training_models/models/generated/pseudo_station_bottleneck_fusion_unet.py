import torch
import torch.nn as nn
import torch.nn.functional as F

class pseudo_station_bottleneck_fusion_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class BottleneckFusion(nn.Module):
        def __init__(self, ch):
            super().__init__()
            hidden = max(ch // 4, 8)
            self.local = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch, kernel_size=3, padding=0, groups=ch, bias=False),
                nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, ch), ch),
                nn.SiLU(inplace=True),
            )
            self.station = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, ch, kernel_size=1),
                nn.Sigmoid(),
            )
            self.mix = nn.Sequential(
                nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, ch), ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            local = self.local(x)
            station = x * self.station(x)
            return self.mix(torch.cat([local, station], dim=1)) + x

    class UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.conv = pseudo_station_bottleneck_fusion_unet.ConvBlock(in_ch + skip_ch, out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.BottleneckFusion(channels[-1])

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self.UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i < len(self.encoders) - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        skips = skips[:-1][::-1]
        for decoder, skip in zip(self.decoders, skips):
            h = decoder(h, skip)

        output = self.head(h)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand(-1, output.shape[1], -1, -1)

        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output