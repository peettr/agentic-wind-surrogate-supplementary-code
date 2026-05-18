import torch
import torch.nn as nn
import torch.nn.functional as F

class coarse_spectral_residual_decoder_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(1, out_ch),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(1, out_ch),
                nn.GELU(),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class SpectralBottleneck(nn.Module):
        def __init__(self, channels, modes=12):
            super().__init__()
            self.modes = modes
            scale = 1.0 / max(1, channels)
            self.weight = nn.Parameter(scale * torch.randn(channels, modes, modes, 2))
            self.mix = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, channels),
                nn.GELU(),
            )

        def forward(self, x):
            b, c, h, w = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros_like(x_ft)

            mh = min(self.modes, h)
            mw = min(self.modes, x_ft.shape[-1])
            wr = torch.view_as_complex(self.weight[:, :mh, :mw].contiguous())
            out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * wr.unsqueeze(0)

            y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
            return x + self.mix(y)

    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [n_c * min(2 ** i, 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            self.ConvBlock(bottleneck_ch, bottleneck_ch),
            self.SpectralBottleneck(bottleneck_ch),
            self.ConvBlock(bottleneck_ch, bottleneck_ch),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(1, channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for i, enc in enumerate(self.encoders):
            y = enc(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels == self.in_channels:
            out_valid = valid
        else:
            out_valid = valid[:, :1].expand(-1, self.out_channels, -1, -1)

        y = torch.where(out_valid, y, torch.full_like(y, float("nan")))
        return y