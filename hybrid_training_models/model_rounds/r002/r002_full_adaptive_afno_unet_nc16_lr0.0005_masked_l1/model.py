import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.block = nn.Sequential(
            ReflectionConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )
        self.skip = nn.Identity() if in_channels == out_channels else ReflectionConv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.block(x) + self.skip(x)

class AdaptiveAFNOBlock(nn.Module):
    def __init__(self, channels, kept_modes=32):
        super().__init__()
        self.channels = channels
        self.kept_modes = kept_modes
        self.real_weight = nn.Parameter(torch.randn(channels, kept_modes, kept_modes) * 0.02)
        self.imag_weight = nn.Parameter(torch.randn(channels, kept_modes, kept_modes) * 0.02)
        self.mix = ReflectionConv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape

        x_ft = torch.fft.rfft2(x, norm="ortho")
        mh = min(self.kept_modes, x_ft.shape[-2])
        mw = min(self.kept_modes, x_ft.shape[-1])

        out_ft = torch.zeros_like(x_ft)
        weight = torch.complex(self.real_weight[:, :mh, :mw], self.imag_weight[:, :mh, :mw])
        out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight.unsqueeze(0)

        x = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        x = self.mix(x)
        x = self.norm(x)
        return residual + self.gate(residual) * x

class adaptive_afno_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoder = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoder.append(ConvBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(2)
        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], channels[-1]),
            AdaptiveAFNOBlock(channels[-1], kept_modes=32),
            ConvBlock(channels[-1], channels[-1]),
        )

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up.append(nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.decoder.append(ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.GELU(),
            ReflectionConv2d(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i != self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up, self.decoder, reversed(skips[:-1])):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))

        output = self.head(h)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != output.shape[1]:
            valid_out = valid_out[:, :1].expand(-1, output.shape[1], -1, -1)

        output = output.clone()
        output[~valid_out] = float("nan")
        return output


