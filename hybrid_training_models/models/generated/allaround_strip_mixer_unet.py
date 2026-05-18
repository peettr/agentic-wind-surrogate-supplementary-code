import torch
import torch.nn as nn
import torch.nn.functional as F

class allaround_strip_mixer_unet(nn.Module):
    class _RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, groups=1):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, groups=groups, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class _Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = allaround_strip_mixer_unet._RefConv(in_ch, out_ch, 3)
            self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.conv2 = allaround_strip_mixer_unet._RefConv(out_ch, out_ch, 3)
            self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)

        def forward(self, x):
            r = self.skip(x)
            x = F.silu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return F.silu(x + r)

    class _StripMixer(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.h_pad = nn.ReflectionPad2d((3, 3, 0, 0))
            self.v_pad = nn.ReflectionPad2d((0, 0, 3, 3))
            self.h = nn.Conv2d(ch, ch, (1, 7), padding=0, groups=ch, bias=False)
            self.v = nn.Conv2d(ch, ch, (7, 1), padding=0, groups=ch, bias=False)
            self.mix = nn.Conv2d(ch, ch, 1, padding=0, bias=False)
            self.norm = nn.GroupNorm(min(8, ch), ch)

        def forward(self, x):
            y = self.h(self.h_pad(x)) + self.v(self.v_pad(x))
            y = self.mix(y)
            return F.silu(self.norm(y) + x)

    class _Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = allaround_strip_mixer_unet._Block(in_ch, out_ch)
            self.strip = allaround_strip_mixer_unet._StripMixer(out_ch)

        def forward(self, x):
            x = self.block(x)
            x = self.strip(x)
            return x

    class _Up(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)
            self.block = allaround_strip_mixer_unet._Block(out_ch + skip_ch, out_ch)
            self.strip = allaround_strip_mixer_unet._StripMixer(out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            x = self.block(x)
            x = self.strip(x)
            return x

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=5):
        super().__init__()
        depth = max(1, int(depth))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self._Block(in_channels, self.channels[0])
        self.encoders = nn.ModuleList()
        for i in range(1, depth):
            self.encoders.append(self._Down(self.channels[i - 1], self.channels[i]))

        self.pool = nn.AvgPool2d(2)
        self.bottleneck = nn.Sequential(
            self._Block(self.channels[-1], self.channels[-1]),
            self._StripMixer(self.channels[-1]),
            self._Block(self.channels[-1], self.channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self._Up(self.channels[i + 1], self.channels[i], self.channels[i]))

        self.head = nn.Sequential(
            self._RefConv(self.channels[0], self.channels[0], 3),
            nn.GroupNorm(min(8, self.channels[0]), self.channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x).any(dim=1, keepdim=True)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for enc in self.encoders:
            y = self.pool(y)
            y = enc(y)
            skips.append(y)

        y = self.bottleneck(y)

        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            y = dec(y, skip)

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return y.masked_fill(~valid.expand(-1, self.out_channels, -1, -1), float("nan"))