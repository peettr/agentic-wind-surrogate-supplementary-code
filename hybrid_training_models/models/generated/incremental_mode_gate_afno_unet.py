import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class incremental_mode_gate_afno_unet(nn.Module):
    class ReflectionConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                incremental_mode_gate_afno_unet.ReflectionConv2d(in_ch, out_ch, 3, bias=False),
                _gn(out_ch),
                nn.GELU(),
                incremental_mode_gate_afno_unet.ReflectionConv2d(out_ch, out_ch, 3, bias=False),
                _gn(out_ch),
                nn.GELU(),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class AFNOBottleneck(nn.Module):
        def __init__(self, channels, modes=16):
            super().__init__()
            self.channels = channels
            self.modes = modes
            self.real_weight = nn.Parameter(torch.randn(channels, modes, modes) * 0.02)
            self.imag_weight = nn.Parameter(torch.randn(channels, modes, modes) * 0.02)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid(),
            )
            self.local = incremental_mode_gate_afno_unet.ConvBlock(channels, channels)

        def forward(self, x):
            b, c, h, w = x.shape
            residual = x
            x_ft = torch.fft.rfft2(x, norm="ortho")

            mh = min(self.modes, x_ft.shape[-2])
            mw = min(self.modes, x_ft.shape[-1])

            low = x_ft[:, :, :mh, :mw]
            weight = torch.complex(
                self.real_weight[:, :mh, :mw],
                self.imag_weight[:, :mh, :mw],
            ).unsqueeze(0)

            x_ft_new = torch.zeros_like(x_ft)
            x_ft_new[:, :, :mh, :mw] = low * weight

            spectral = torch.fft.irfft2(x_ft_new, s=(h, w), norm="ortho")
            gate = self.gate(residual)
            return self.local(residual + gate * spectral)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.out_channels = out_channels
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.AvgPool2d(2)
        self.bottleneck = self.AFNOBottleneck(channels[-1], modes=16)

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(channels[i + 1], channels[i], 1, bias=False))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i < len(self.encoders) - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1))

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            if valid.shape[1] == 1:
                valid = valid.expand(-1, output.shape[1], -1, -1)
            else:
                valid = valid.all(dim=1, keepdim=True).expand(-1, output.shape[1], -1, -1)

        return torch.where(valid, output, torch.full_like(output, float("nan")))


