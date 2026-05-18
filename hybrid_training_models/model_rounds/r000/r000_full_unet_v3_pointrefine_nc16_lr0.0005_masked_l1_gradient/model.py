import torch
import torch.nn as nn
import torch.nn.functional as F


class unet_v3_pointrefine(nn.Module):
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

    class UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
            self.conv = unet_v3_pointrefine.ConvBlock(out_ch + skip_ch, out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        max_ch = n_c * 8
        channels = [min(n_c * (2 ** i), max_ch) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.AvgPool2d(2)
        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        curr_ch = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            out_ch = skip_ch
            self.decoders.append(self.UpBlock(curr_ch, skip_ch, out_ch))
            curr_ch = out_ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(curr_ch, curr_ch, 3, bias=False),
            nn.GroupNorm(min(8, curr_ch), curr_ch),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(curr_ch, out_channels, 3),
        )

        self.refine = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels + in_channels, n_c, 3, bias=False),
            nn.GroupNorm(min(8, n_c), n_c),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_c, out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            h = dec(h, skip)

        output = self.head(h)
        output = output + self.refine(torch.cat([output, x_masked], dim=1))

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        output = output.masked_fill(~valid, float("nan"))
        return output


