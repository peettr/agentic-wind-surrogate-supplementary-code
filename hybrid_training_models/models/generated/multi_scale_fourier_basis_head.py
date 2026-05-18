import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for num_groups in range(min(8, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class multi_scale_fourier_basis_head(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, groups=1):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, groups=groups, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = multi_scale_fourier_basis_head.ReflectConv(in_ch, out_ch, 3)
            self.norm1 = _gn(out_ch)
            self.conv2 = multi_scale_fourier_basis_head.ReflectConv(out_ch, out_ch, 3)
            self.norm2 = _gn(out_ch)
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)

        def forward(self, x):
            y = F.silu(self.norm1(self.conv1(x)))
            y = self.norm2(self.conv2(y))
            return F.silu(y + self.skip(x))

    class FourierBlock(nn.Module):
        def __init__(self, channels, modes_h=24, modes_w=24):
            super().__init__()
            self.modes_h = modes_h
            self.modes_w = modes_w
            self.real = nn.Parameter(torch.randn(channels, modes_h, modes_w) * 0.02)
            self.imag = nn.Parameter(torch.randn(channels, modes_h, modes_w) * 0.02)
            self.local = multi_scale_fourier_basis_head.ResBlock(channels, channels)

        def forward(self, x):
            b, c, h, w = x.shape
            mh = min(self.modes_h, h)
            mw = min(self.modes_w, w // 2 + 1)

            xf = torch.fft.rfft2(x, norm="ortho")
            yf = torch.zeros_like(xf)

            weight = torch.complex(self.real[:, :mh, :mw], self.imag[:, :mh, :mw])
            yf[:, :, :mh, :mw] = xf[:, :, :mh, :mw] * weight.unsqueeze(0)

            y = torch.fft.irfft2(yf, s=(h, w), norm="ortho")
            return self.local(x + y)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.ResBlock(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        for i in range(1, depth):
            self.encoders.append(self.ResBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            self.FourierBlock(channels[-1]),
            self.ResBlock(channels[-1], channels[-1])
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self.ResBlock(channels[i + 1] + channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ResBlock(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for enc in self.encoders:
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
            y = enc(y)
            skips.append(y)

        y = self.bottleneck(y)

        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid[:, :1].expand(-1, y.shape[1], -1, -1)

        return torch.where(valid, y, torch.full_like(y, float("nan")))