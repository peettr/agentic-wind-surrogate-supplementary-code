import torch
import torch.nn as nn
import torch.nn.functional as F


class unet_sdf_7level(nn.Module):
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

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.up = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(channels[:-1]):
            self.up.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(prev_ch, ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, ch), ch),
                nn.SiLU(inplace=True),
            ))
            self.decoders.append(self.ConvBlock(ch * 2, ch))
            prev_ch = ch

        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        skips = skips[:-1][::-1]
        for up, dec, skip in zip(self.up, self.decoders, skips):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        output = self.out(h)
        output_valid = valid if valid.shape[1] == output.shape[1] else valid.all(dim=1, keepdim=True)
        output = torch.where(output_valid, output, torch.full_like(output, float("nan")))
        return output


