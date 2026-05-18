import torch
import torch.nn as nn
import torch.nn.functional as F

class afno_block(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class _SpectralBottleneck(nn.Module):
        def __init__(self, channels, modes=20):
            super().__init__()
            self.channels = channels
            self.modes = modes
            scale = 1.0 / (channels ** 0.5)
            self.weight_real = nn.Parameter(scale * torch.randn(channels, modes, modes))
            self.weight_imag = nn.Parameter(scale * torch.randn(channels, modes, modes))
            self.mix = nn.Sequential(
                nn.Conv2d(channels, channels * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(channels * 2, channels, kernel_size=1),
            )
            self.norm = nn.GroupNorm(min(8, channels), channels)

        def forward(self, x):
            b, c, h, w = x.shape
            mh = min(self.modes, h)
            mw = min(self.modes, w // 2 + 1)

            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros_like(x_ft)

            weight = torch.complex(
                self.weight_real[:, :mh, :mw],
                self.weight_imag[:, :mh, :mw],
            )
            out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight.unsqueeze(0)

            y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
            y = self.mix(y)
            return self.norm(x + y)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=5):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            self._ConvBlock(channels[-1], channels[-1]),
            self._SpectralBottleneck(channels[-1]),
            self._ConvBlock(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoders.append(self._ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i != len(self.encoders) - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = proj(y)
            y = decoder(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != y.shape[1]:
            valid_out = valid_out.expand(-1, y.shape[1], -1, -1)
        y = torch.where(valid_out, y, torch.full_like(y, float("nan")))
        return y