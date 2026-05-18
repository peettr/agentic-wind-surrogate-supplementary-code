import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _RefConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _RefConv2d(in_channels, out_channels, 3),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            _RefConv2d(out_channels, out_channels, 3),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _FourierMix(nn.Module):
    def __init__(self, channels, modes=24):
        super().__init__()
        self.channels = channels
        self.modes = modes
        self.real_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.imag_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)

        xf = torch.fft.rfft2(x, norm="ortho")
        low = xf[:, :, :mh, :mw]

        weight = torch.complex(self.real_scale, self.imag_scale)
        low = low * weight

        out_f = torch.zeros_like(xf)
        out_f[:, :, :mh, :mw] = low

        out = torch.fft.irfft2(out_f, s=(h, w), norm="ortho")
        return x + self.proj(out)


class separable_fourier_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        depth = max(1, int(depth))
        n_c = max(1, int(n_c))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(_ConvBlock(prev, ch))
            prev = ch

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            _ConvBlock(channels[-1], channels[-1]),
            _FourierMix(channels[-1]),
            _ConvBlock(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for skip_ch in reversed(channels):
            self.up_projs.append(nn.Conv2d(prev, skip_ch, 1))
            self.decoders.append(_ConvBlock(skip_ch * 2, skip_ch))
            prev = skip_ch

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=0)

    def forward(self, x):
        invalid = torch.isnan(x)
        x_masked = torch.where(invalid, torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i < len(self.encoders) - 1 and min(h.shape[-2:]) >= 4:
                h = self.pool(h)

        h = self.bottleneck(h)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        output = self.out_conv(self.out_pad(h))

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if invalid.shape[1] == output.shape[1]:
            output_invalid = invalid
        else:
            output_invalid = invalid.any(dim=1, keepdim=True).expand_as(output)

        output = output.masked_fill(output_invalid, float("nan"))
        return output