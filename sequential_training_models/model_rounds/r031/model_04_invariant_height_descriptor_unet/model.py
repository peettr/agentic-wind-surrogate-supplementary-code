import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        g = min(8, out_channels)
        while out_channels % g != 0:
            g -= 1

        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.GroupNorm(g, out_channels),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
            nn.GroupNorm(g, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class invariant_height_descriptor_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(_ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = _ConvBlock(channels[-1], channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.decoders.append(_ConvBlock(channels[i] + channels[i], channels[i]))

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3)

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

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, skips):
            h = upconv(h)

            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        h = self.out_conv(self.out_pad(h))

        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != h.shape[1]:
            valid = valid[:, :1].expand_as(h)

        return torch.where(valid, h, torch.full_like(h, float("nan")))


