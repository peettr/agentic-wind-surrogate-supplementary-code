import torch
import torch.nn as nn
import torch.nn.functional as F

class axis_factor_spectral_adapter_unet(nn.Module):
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
                axis_factor_spectral_adapter_unet.ReflectionConv2d(in_channels, out_channels, 3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                axis_factor_spectral_adapter_unet.ReflectionConv2d(out_channels, out_channels, 3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class AxisFactorSpectralAdapter(nn.Module):
        def __init__(self, channels, modes_h=24, modes_w=24):
            super().__init__()
            self.channels = channels
            self.modes_h = modes_h
            self.modes_w = modes_w

            scale = 1.0 / max(1, channels)
            self.row_real = nn.Parameter(scale * torch.randn(channels, modes_w))
            self.row_imag = nn.Parameter(scale * torch.randn(channels, modes_w))
            self.col_real = nn.Parameter(scale * torch.randn(channels, modes_h))
            self.col_imag = nn.Parameter(scale * torch.randn(channels, modes_h))

            self.mix = nn.Sequential(
                nn.Conv2d(channels, channels, 1, padding=0, bias=False),
                nn.GroupNorm(min(8, channels), channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, 1, padding=0, bias=False),
            )
            self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x):
            b, c, h, w = x.shape
            dtype = x.dtype

            x_float = x.float()
            spec = torch.fft.rfft2(x_float, norm="ortho")
            out_spec = torch.zeros_like(spec)

            mh = min(self.modes_h, h)
            mw = min(self.modes_w, spec.shape[-1])

            row_weight = torch.complex(
                self.row_real[:, :mw].float(),
                self.row_imag[:, :mw].float()
            ).view(1, c, 1, mw)

            col_weight = torch.complex(
                self.col_real[:, :mh].float(),
                self.col_imag[:, :mh].float()
            ).view(1, c, mh, 1)

            weight = col_weight * row_weight
            out_spec[:, :, :mh, :mw] = spec[:, :, :mh, :mw] * weight

            y = torch.fft.irfft2(out_spec, s=(h, w), norm="ortho").to(dtype)
            y = self.mix(y)
            return x + torch.tanh(self.gate) * y

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_proj = self.ReflectionConv2d(in_channels, channels[0], 3, bias=False)

        self.encoder = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.encoder.append(self.ConvBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.AxisFactorSpectralAdapter(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.up_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_proj.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0, bias=False))
            self.decoder.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.output_head = nn.Sequential(
            self.ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
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

        for up, block, skip in zip(self.up_proj, self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(y)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        y = self.output_head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.expand(-1, y.shape[1], -1, -1)

        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y