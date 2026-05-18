import torch
import torch.nn as nn
import torch.nn.functional as F

class vq_pressure_regime_codebook_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        max_ch = n_c * 8
        channels = [min(n_c * (2 ** i), max_ch) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.out_conv(self.out_pad(h))

        if valid.shape != out.shape:
            valid = valid.all(dim=1, keepdim=True).expand_as(out)

        out = out.clone()
        out[~valid] = float("nan")
        return out