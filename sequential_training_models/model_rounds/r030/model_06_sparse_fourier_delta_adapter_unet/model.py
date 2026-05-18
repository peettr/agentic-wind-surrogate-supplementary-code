import torch
import torch.nn as nn
import torch.nn.functional as F

class sparse_fourier_delta_adapter_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                sparse_fourier_delta_adapter_unet.ReflectConv(in_channels, out_channels, 3),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
                sparse_fourier_delta_adapter_unet.ReflectConv(out_channels, out_channels, 3),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class FourierDeltaAdapter(nn.Module):
        def __init__(self, channels, modes=16):
            super().__init__()
            self.modes = modes
            self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.weight_real = nn.Parameter(0.01 * torch.randn(1, channels, modes, modes))
            self.weight_imag = nn.Parameter(0.01 * torch.randn(1, channels, modes, modes))

        def forward(self, x):
            b, c, h, w = x.shape
            mh = min(self.modes, h)
            mw = min(self.modes, w // 2 + 1)

            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros_like(x_ft)

            weight = torch.complex(
                self.weight_real[:, :, :mh, :mw],
                self.weight_imag[:, :, :mh, :mw],
            )
            out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight

            delta = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
            return x + self.scale * delta

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.FourierDeltaAdapter(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            if i != len(self.encoders) - 1:
                skips.append(y)
                y = self.pool(y)

        y = self.bottleneck(y)

        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            y = up(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = decoder(y)

        y = self.out_conv(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return torch.where(valid, y, torch.full_like(y, float("nan")))


