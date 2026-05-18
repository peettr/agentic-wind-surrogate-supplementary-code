import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class warped_anisotropic_boundary_hybrid_unet(nn.Module):
    class ReflectionConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, bias=False):
            super().__init__()
            if isinstance(kernel_size, int):
                kh, kw = kernel_size, kernel_size
            else:
                kh, kw = kernel_size
            self.pad = nn.ReflectionPad2d((kw // 2, kw // 2, kh // 2, kh // 2))
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kh, kw), bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            mid_channels = out_channels
            self.main = nn.Sequential(
                warped_anisotropic_boundary_hybrid_unet.ReflectionConv2d(in_channels, mid_channels, 3),
                _gn(mid_channels),
                nn.SiLU(inplace=True),
                warped_anisotropic_boundary_hybrid_unet.ReflectionConv2d(mid_channels, mid_channels, (1, 5)),
                _gn(mid_channels),
                nn.SiLU(inplace=True),
                warped_anisotropic_boundary_hybrid_unet.ReflectionConv2d(mid_channels, out_channels, (5, 1)),
                _gn(out_channels),
            )
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.main(x) + self.skip(x))

    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = warped_anisotropic_boundary_hybrid_unet.ConvBlock(in_channels, out_channels)

        def forward(self, x):
            y = self.block(x)
            d = F.avg_pool2d(y, kernel_size=2, stride=2)
            return y, d

    class UpBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.block = warped_anisotropic_boundary_hybrid_unet.ConvBlock(out_channels + skip_channels, out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            return self.block(torch.cat([x, skip], dim=1))

    class Bottleneck(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.local = warped_anisotropic_boundary_hybrid_unet.ConvBlock(channels, channels)
            self.mix = nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                _gn(channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            y = self.local(x)
            fy = torch.fft.rfft2(y, norm="ortho")
            h_cut = min(24, fy.shape[-2])
            w_cut = min(24, fy.shape[-1])
            low = torch.zeros_like(fy)
            low[:, :, :h_cut, :w_cut] = fy[:, :, :h_cut, :w_cut]
            low = torch.fft.irfft2(low, s=y.shape[-2:], norm="ortho")
            return y + self.mix(low)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels[:-1]:
            self.encoders.append(self.DownBlock(prev, ch))
            prev = ch

        self.bottom_in = self.ConvBlock(prev, channels[-1])
        self.bottleneck = self.Bottleneck(channels[-1])

        self.decoders = nn.ModuleList()
        prev = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.decoders.append(self.UpBlock(prev, skip_ch, skip_ch))
            prev = skip_ch

        self.head = nn.Sequential(
            self.ReflectionConv2d(prev, prev, 3),
            _gn(prev),
            nn.SiLU(inplace=True),
            nn.Conv2d(prev, out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked
        for encoder in self.encoders:
            skip, y = encoder(y)
            skips.append(skip)

        y = self.bottom_in(y)
        y = self.bottleneck(y)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            y = decoder(y, skip)

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != y.shape[1]:
            valid_out = valid_out[:, :1].expand(-1, y.shape[1], -1, -1)
        y = torch.where(valid_out, y, torch.full_like(y, float("nan")))
        return y