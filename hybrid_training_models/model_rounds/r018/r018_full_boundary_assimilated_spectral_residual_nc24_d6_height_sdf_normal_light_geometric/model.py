import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _ReflectConv2d(channels, channels, 3)
        self.norm1 = _gn(channels)
        self.conv2 = _ReflectConv2d(channels, channels, 3)
        self.norm2 = _gn(channels)

    def forward(self, x):
        r = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + r)


class _Stage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = _ReflectConv2d(in_channels, out_channels, 3)
        self.norm = _gn(out_channels)
        self.res = _ResBlock(out_channels)

    def forward(self, x):
        x = F.silu(self.norm(self.proj(x)))
        return self.res(x)


class _SpectralResidual(nn.Module):
    def __init__(self, channels, modes=16):
        super().__init__()
        self.channels = channels
        self.modes = modes
        self.real = nn.Parameter(torch.randn(channels, modes, modes) * 0.02)
        self.imag = nn.Parameter(torch.randn(channels, modes, modes) * 0.02)
        self.mix = _ReflectConv2d(channels, channels, 1)
        self.norm = _gn(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)

        xf = torch.fft.rfft2(x, norm="ortho")
        yf = torch.zeros_like(xf)
        weight = torch.complex(self.real[:, :mh, :mw], self.imag[:, :mh, :mw])
        yf[:, :, :mh, :mw] = xf[:, :, :mh, :mw] * weight.unsqueeze(0)

        y = torch.fft.irfft2(yf, s=(h, w), norm="ortho")
        y = self.mix(y)
        return F.silu(self.norm(x + y))


class boundary_assimilated_spectral_residual(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        depth = max(1, int(depth))
        self.in_channels = in_channels
        self.out_channels = out_channels
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.input = _Stage(in_channels + 1, channels[0])
        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(_Stage(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            _ResBlock(channels[-1]),
            _SpectralResidual(channels[-1], modes=16),
            _ResBlock(channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(_Stage(channels[i + 1] + channels[i], channels[i]))

        self.output = nn.Sequential(
            _ResBlock(channels[0]),
            _ReflectConv2d(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))
        mask = valid.any(dim=1, keepdim=True).to(dtype=x_masked.dtype)

        y = torch.cat([x_masked, mask], dim=1)
        skips = []

        y = self.input(y)
        skips.append(y)

        for block in self.encoder:
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        y = self.output(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        spatial_valid = valid.any(dim=1, keepdim=True)
        y = torch.where(spatial_valid, y, torch.full_like(y, float("nan")))
        return y


