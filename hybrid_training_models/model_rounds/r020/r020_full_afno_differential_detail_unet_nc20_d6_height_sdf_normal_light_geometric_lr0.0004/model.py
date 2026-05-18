import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class afno_differential_detail_unet(nn.Module):
    class ReflectionConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
            super().__init__()
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                afno_differential_detail_unet.ReflectionConv2d(in_channels, out_channels, 3),
                _gn(out_channels),
                nn.SiLU(inplace=True),
                afno_differential_detail_unet.ReflectionConv2d(out_channels, out_channels, 3),
                _gn(out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class AFNOBlock(nn.Module):
        def __init__(self, channels, modes=16):
            super().__init__()
            self.channels = channels
            self.modes = modes
            scale = 1.0 / max(1, channels)
            self.w_real = nn.Parameter(scale * torch.randn(channels, modes, modes))
            self.w_imag = nn.Parameter(scale * torch.randn(channels, modes, modes))
            self.mix = nn.Sequential(
                afno_differential_detail_unet.ReflectionConv2d(channels, channels, 1),
                _gn(channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            b, c, h, w = x.shape
            residual = x
            x_ft = torch.fft.rfft2(x, norm="ortho")
            mh = min(self.modes, x_ft.shape[-2])
            mw = min(self.modes, x_ft.shape[-1])

            out_ft = x_ft.clone()
            xr = x_ft[:, :, :mh, :mw].real
            xi = x_ft[:, :, :mh, :mw].imag
            wr = self.w_real[:, :mh, :mw].unsqueeze(0)
            wi = self.w_imag[:, :mh, :mw].unsqueeze(0)

            out_ft[:, :, :mh, :mw] = torch.complex(xr * wr - xi * wi, xr * wi + xi * wr)
            x = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
            return residual + self.mix(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        prev = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev, ch))
            self.downs.append(nn.AvgPool2d(2))
            prev = ch

        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            self.ConvBlock(bottleneck_channels, bottleneck_channels),
            self.AFNOBlock(bottleneck_channels, modes=16),
            self.ConvBlock(bottleneck_channels, bottleneck_channels),
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for skip_ch in reversed(channels):
            self.up_convs.append(self.ReflectionConv2d(prev, skip_ch, 1))
            self.decoders.append(self.ConvBlock(skip_ch * 2, skip_ch))
            prev = skip_ch

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for encoder, down in zip(self.encoders, self.downs):
            y = encoder(y)
            skips.append(y)
            y = down(y)

        y = self.bottleneck(y)

        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, reversed(skips)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_conv(y)
            y = decoder(torch.cat([y, skip], dim=1))

        y = self.out_conv(self.out_pad(y))

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(y)

        return torch.where(valid, y, torch.full_like(y, float("nan")))


