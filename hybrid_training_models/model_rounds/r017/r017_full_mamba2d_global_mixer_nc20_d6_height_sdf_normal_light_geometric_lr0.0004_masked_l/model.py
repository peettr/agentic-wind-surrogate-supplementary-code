import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class mamba2d_global_mixer(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class GlobalMixBlock(nn.Module):
        def __init__(self, channels, modes=32):
            super().__init__()
            self.channels = channels
            self.modes = modes
            self.norm = _gn(channels)
            self.local = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, 3, groups=channels, bias=False),
                nn.Conv2d(channels, channels * 2, 1),
            )
            self.freq_real = nn.Parameter(torch.randn(channels, modes, modes) * 0.02)
            self.freq_imag = nn.Parameter(torch.randn(channels, modes, modes) * 0.02)
            self.proj = nn.Conv2d(channels, channels, 1)

        def forward(self, x):
            b, c, h, w = x.shape
            z = self.norm(x)

            local, gate = self.local(z).chunk(2, dim=1)
            local = local * torch.sigmoid(gate)

            xf = torch.fft.rfft2(z, norm="ortho")
            mh = min(self.modes, xf.shape[-2])
            mw = min(self.modes, xf.shape[-1])

            weight = torch.complex(
                self.freq_real[:, :mh, :mw],
                self.freq_imag[:, :mh, :mw],
            )

            yf = torch.zeros_like(xf)
            yf[:, :, :mh, :mw] = xf[:, :, :mh, :mw] * weight.unsqueeze(0)
            global_part = torch.fft.irfft2(yf, s=(h, w), norm="ortho")

            return x + self.proj(local + global_part)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stem = self.ConvBlock(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        for i in range(1, depth):
            self.encoders.append(self.ConvBlock(channels[i - 1], channels[i]))

        self.down = nn.AvgPool2d(2)

        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            self.ConvBlock(bottleneck_ch, bottleneck_ch),
            self.GlobalMixBlock(bottleneck_ch),
            self.GlobalMixBlock(bottleneck_ch),
            self.ConvBlock(bottleneck_ch, bottleneck_ch),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1, bias=False))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for enc in self.encoders:
            y = self.down(y)
            y = enc(y)
            skips.append(y)

        y = self.bottleneck(y)

        for proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = proj(y)
            y = torch.cat([y, skip], dim=1)
            y = dec(y)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.any(dim=1, keepdim=True).expand_as(y)

        return torch.where(valid, y, torch.full_like(y, float("nan")))