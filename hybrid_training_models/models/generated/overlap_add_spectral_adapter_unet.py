import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class overlap_add_spectral_adapter_unet(nn.Module):
    class ReflectionConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                overlap_add_spectral_adapter_unet.ReflectionConv2d(in_channels, out_channels, 3, bias=False),
                _gn(out_channels),
                nn.SiLU(inplace=True),
                overlap_add_spectral_adapter_unet.ReflectionConv2d(out_channels, out_channels, 3, bias=False),
                _gn(out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class SpectralAdapter(nn.Module):
        def __init__(self, channels, modes=16):
            super().__init__()
            self.channels = channels
            self.modes = modes
            self.real = nn.Parameter(torch.randn(channels, modes, modes) * 0.02)
            self.imag = nn.Parameter(torch.randn(channels, modes, modes) * 0.02)
            self.mix = nn.Conv2d(channels, channels, 1, padding=0, bias=True)

        def forward(self, x):
            b, c, h, w = x.shape
            modes_h = min(self.modes, h)
            modes_w = min(self.modes, w // 2 + 1)

            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros_like(x_ft)

            weight = torch.complex(
                self.real[:, :modes_h, :modes_w],
                self.imag[:, :modes_h, :modes_w],
            )
            out_ft[:, :, :modes_h, :modes_w] = x_ft[:, :, :modes_h, :modes_w] * weight.unsqueeze(0)

            y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
            return x + self.mix(y)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_proj = self.ReflectionConv2d(in_channels, channels[0], 3, bias=True)

        self.encoder = nn.ModuleList()
        prev_channels = channels[0]
        for ch in channels:
            self.encoder.append(self.ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.SpectralAdapter(channels[-1], modes=16),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self.ConvBlock(channels[i + 1] + channels[i], channels[i]))

        self.output_head = nn.Sequential(
            self.ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            self.ReflectionConv2d(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        y = self.input_proj(x_masked)

        skips = []
        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i != self.depth - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        y = self.output_head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)
        else:
            valid_out = valid

        nan_fill = torch.full_like(y, float("nan"))
        y = torch.where(valid_out, y, nan_fill)
        return y