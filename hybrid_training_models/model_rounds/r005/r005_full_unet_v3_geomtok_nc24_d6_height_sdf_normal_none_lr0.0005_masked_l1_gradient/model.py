import torch
import torch.nn as nn
import torch.nn.functional as F

class unet_v3_geomtok(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.AvgPool2d(2)

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.up_reduce = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_reduce.append(
                nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(channels[i + 1], channels[i], 3, bias=False),
                    nn.GroupNorm(min(8, channels[i]), channels[i]),
                    nn.SiLU(inplace=True),
                )
            )
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.out_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3),
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
                h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_reduce, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.out_head(h)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels == self.in_channels:
            out = torch.where(valid, out, torch.full_like(out, float("nan")))
        else:
            out_valid = valid[:, :1].expand(-1, self.out_channels, -1, -1)
            out = torch.where(out_valid, out, torch.full_like(out, float("nan")))

        return out