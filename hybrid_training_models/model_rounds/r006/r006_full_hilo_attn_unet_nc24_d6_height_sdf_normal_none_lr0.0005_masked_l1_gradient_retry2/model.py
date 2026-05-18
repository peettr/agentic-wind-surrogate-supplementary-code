import torch
import torch.nn as nn
import torch.nn.functional as F


class hilo_attn_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=False)
            self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.norm(self.conv(self.pad(x))))

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = hilo_attn_unet.RefConv(in_ch, out_ch)
            self.conv2 = hilo_attn_unet.RefConv(out_ch, out_ch)
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)

        def forward(self, x):
            return self.conv2(self.conv1(x)) + self.skip(x)

    class HiLoAttn(nn.Module):
        def __init__(self, ch):
            super().__init__()
            hidden = max(ch // 4, 8)
            self.local = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch, 3, padding=0, groups=ch, bias=False),
                nn.Conv2d(ch, ch, 1, padding=0, bias=False),
                nn.GroupNorm(min(8, ch), ch),
                nn.SiLU(inplace=True),
            )
            self.low = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, hidden, 1, padding=0),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, ch, 1, padding=0),
                nn.Sigmoid(),
            )
            self.mix = nn.Conv2d(ch, ch, 1, padding=0, bias=False)

        def forward(self, x):
            return x + self.mix(self.local(x) * self.low(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(nn.Sequential(
                self.Block(prev, ch),
                self.HiLoAttn(ch),
            ))
            prev = ch

        self.down = nn.AvgPool2d(2)

        self.bottleneck = nn.Sequential(
            self.Block(channels[-1], channels[-1]),
            self.HiLoAttn(channels[-1]),
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0, bias=False))
            self.decoders.append(nn.Sequential(
                self.Block(channels[i] * 2, channels[i]),
                self.HiLoAttn(channels[i]),
            ))

        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, padding=0, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        nan_mask = torch.isnan(x)
        valid = ~nan_mask.any(dim=1, keepdim=True)
        x_masked = torch.where(nan_mask, torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1))

        output = self.out(h)
        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output


