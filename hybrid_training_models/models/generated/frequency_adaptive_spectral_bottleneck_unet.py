import torch
import torch.nn as nn
import torch.nn.functional as F


class frequency_adaptive_spectral_bottleneck_unet(nn.Module):
    @staticmethod
    def _gn(num_channels):
        groups = min(8, num_channels)
        while num_channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)

    class _ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False),
                frequency_adaptive_spectral_bottleneck_unet._gn(out_channels),
                nn.GELU()
            )

        def forward(self, x):
            return self.net(x)

    class _Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                frequency_adaptive_spectral_bottleneck_unet._ReflectConv(in_channels, out_channels),
                frequency_adaptive_spectral_bottleneck_unet._ReflectConv(out_channels, out_channels)
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
            )

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class _SpectralBottleneck(nn.Module):
        def __init__(self, channels, modes=16):
            super().__init__()
            self.modes = modes
            scale = 1.0 / max(1, channels)
            self.weight_real = nn.Parameter(scale * torch.randn(channels, modes, modes))
            self.weight_imag = nn.Parameter(scale * torch.randn(channels, modes, modes))
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(1, channels // 4), kernel_size=1),
                nn.GELU(),
                nn.Conv2d(max(1, channels // 4), channels, kernel_size=1),
                nn.Sigmoid()
            )
            self.local = frequency_adaptive_spectral_bottleneck_unet._Block(channels, channels)

        def forward(self, x):
            b, c, h, w = x.shape
            mh = min(self.modes, h)
            mw = min(self.modes, w // 2 + 1)

            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros_like(x_ft)

            weight = torch.complex(
                self.weight_real[:, :mh, :mw],
                self.weight_imag[:, :mh, :mw]
            ).to(dtype=x_ft.dtype, device=x.device)

            gate = self.gate(x).to(dtype=x_ft.real.dtype)
            out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight.unsqueeze(0)
            spectral = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")

            return self.local(x) + spectral * gate

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self._Block(prev_channels, ch))
            prev_channels = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._SpectralBottleneck(channels[-1], modes=16)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.decoders.append(self._Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self._ReflectConv(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips[:-1])):
            h = upconv(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = decoder(torch.cat([h, skip], dim=1))

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid[:, :1].expand(-1, out.shape[1], -1, -1)
        return torch.where(out_valid, out, torch.full_like(out, float("nan")))