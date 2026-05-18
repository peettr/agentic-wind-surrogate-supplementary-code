import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierBlock(nn.Module):
    def __init__(self, channels, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(channels, modes, modes, 2) * 0.02)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        b, c, h, w = x.shape
        m1 = min(self.modes, h)
        m2 = min(self.modes, w // 2 + 1)

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros_like(x_ft)

        weight = torch.view_as_complex(self.weight[:, :m1, :m2].contiguous())
        out_ft[:, :, :m1, :m2] = x_ft[:, :, :m1, :m2] * weight.unsqueeze(0)

        y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        y = self.mix(y)
        return F.gelu(self.norm(x + y))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)

        self.fourier = FourierBlock(out_ch, modes)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(self.pad1(x))))
        x = F.gelu(self.norm2(self.conv2(self.pad2(x))))
        x = self.fourier(x)
        return x + residual


class p8_fourier_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7, modes=12):
        super().__init__()
        # CRITICAL: total params must be under 50M.
        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), 512) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.encoders = nn.ModuleList()
        prev = in_channels + 1
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch, modes))
            prev = ch

        self.pool = nn.AvgPool2d(kernel_size=2)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.ups.append(nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(channels[i] * 2, channels[i], modes))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        nan_mask = torch.isnan(x)
        valid = (~nan_mask).to(dtype=x.dtype)
        x = torch.where(nan_mask, torch.zeros_like(x), x)
        x = torch.cat([x, valid[:, :1]], dim=1)

        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if i != len(self.encoders) - 1:
                x = self.pool(x)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        x = self.head(x)
        return x