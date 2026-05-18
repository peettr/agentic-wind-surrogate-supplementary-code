import torch
import torch.nn as nn
import torch.nn.functional as F

class unet_v3_geomtok(nn.Module):
    class _ConvBlock(nn.Module):
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

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        max_c = n_c * 8
        channels = [min(n_c * (2 ** i), max_c) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_c = in_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(prev_c, ch))
            prev_c = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._ConvBlock(channels[-1], channels[-1])

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoders.append(self._ConvBlock(channels[i] * 2, channels[i]))

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        h = self.out_conv(self.out_pad(h))

        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if h.shape[1] != valid_out.shape[1]:
            valid_out = valid_out[:, :1].expand(-1, h.shape[1], -1, -1)

        return torch.where(valid_out, h, torch.full_like(h, float("nan")))


