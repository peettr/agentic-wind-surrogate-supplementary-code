import torch
import torch.nn as nn
import torch.nn.functional as F

class liif_head_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoder.append(self._ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        bottleneck_channels = channels[-1]
        self.bottleneck = self._ConvBlock(channels[-1], bottleneck_channels)

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        dec_in = bottleneck_channels
        for skip_ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(dec_in, skip_ch, kernel_size=2, stride=2))
            self.decoder.append(self._ConvBlock(skip_ch + skip_ch, skip_ch))
            dec_in = skip_ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for block in self.encoder:
            h = block(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, block, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            h = up(h)

            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            h = torch.cat([h, skip], dim=1)
            h = block(h)

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output


