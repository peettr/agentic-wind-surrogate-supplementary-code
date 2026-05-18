import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            _ReflectConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            _ReflectConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Identity() if in_channels == out_channels else _ReflectConv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.block(x) + self.proj(x)

class _SpectralFusion(nn.Module):
    def __init__(self, channels, modes=16):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(channels, modes, modes, 2) * 0.02)
        self.mix = nn.Sequential(
            _ReflectConv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
            _ReflectConv2d(channels, channels, 3, bias=False),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        mh = min(self.modes, h)
        mw = min(self.modes, x_ft.shape[-1])

        out_ft = torch.zeros_like(x_ft)
        weight = torch.view_as_complex(self.weight[:, :mh, :mw].contiguous())
        out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight.unsqueeze(0)

        spectral = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return x + self.mix(torch.cat([x, spectral], dim=1))

class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))

class coarse_to_fine_spectral_fusion_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = _ConvBlock(in_channels, channels[0])
        self.encoders = nn.ModuleList([
            _ConvBlock(channels[i - 1], channels[i]) for i in range(1, depth)
        ])
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            _ConvBlock(channels[-1], channels[-1]),
            _SpectralFusion(channels[-1], modes=16),
            _ConvBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList([
            _UpBlock(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        ])

        self.head = nn.Sequential(
            _ReflectConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            _ReflectConv2d(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for encoder in self.encoders:
            y = self.down(y)
            y = encoder(y)
            skips.append(y)

        y = self.bottleneck(y)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            y = decoder(y, skip)

        y = self.head(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False) if y.shape[-2:] != x.shape[-2:] else y

        out_valid = valid
        if out_valid.shape[1] != y.shape[1]:
            out_valid = out_valid[:, :1].expand(-1, y.shape[1], -1, -1)
        return torch.where(out_valid, y, torch.full_like(y, float("nan")))