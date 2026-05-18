import torch
import torch.nn as nn
import torch.nn.functional as F

class p8_unet_v3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.depth = depth
        channels = [n_c * (2 ** i) for i in range(depth)]

        def conv3x3(in_ch, out_ch):
            return nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def block(in_ch, out_ch):
            return nn.Sequential(
                conv3x3(in_ch, out_ch),
                conv3x3(out_ch, out_ch),
            )

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(block(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = block(channels[-1], channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.decoder.append(block(channels[i] * 2, channels[i]))

        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, enc in enumerate(self.encoder):
            h = enc(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips[:-1])):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        output = self.final(h)
        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output