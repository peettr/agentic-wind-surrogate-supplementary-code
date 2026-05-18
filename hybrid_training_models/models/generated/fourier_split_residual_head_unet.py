import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    groups = min(8, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class fourier_split_residual_head_unet(nn.Module):
    class RefConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = fourier_split_residual_head_unet.RefConv2d(in_channels, out_channels, 3)
            self.norm1 = _gn(out_channels)
            self.conv2 = fourier_split_residual_head_unet.RefConv2d(out_channels, out_channels, 3)
            self.norm2 = _gn(out_channels)
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        def forward(self, x):
            residual = self.skip(x)
            x = F.gelu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return F.gelu(x + residual)

    class FourierBlock(nn.Module):
        def __init__(self, channels, modes=24):
            super().__init__()
            self.channels = channels
            self.modes = modes
            scale = 1.0 / max(1, channels)
            self.weight_real = nn.Parameter(scale * torch.randn(channels, modes, modes))
            self.weight_imag = nn.Parameter(scale * torch.randn(channels, modes, modes))
            self.mix = nn.Conv2d(channels, channels, 1)
            self.norm = _gn(channels)

        def forward(self, x):
            b, c, h, w = x.shape
            mh = min(self.modes, h)
            mw = min(self.modes, w // 2 + 1)

            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros_like(x_ft)

            weight = torch.complex(
                self.weight_real[:, :mh, :mw],
                self.weight_imag[:, :mh, :mw],
            ).unsqueeze(0)

            out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight
            y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
            y = self.mix(y)
            return F.gelu(self.norm(x + y))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self.ResBlock(prev, ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            self.ResBlock(channels[-1], channels[-1]),
            self.FourierBlock(channels[-1]),
            self.ResBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        prev = channels[-1]
        for ch in reversed(channels[:-1]):
            self.up_projs.append(nn.Conv2d(prev, ch, 1))
            self.decoders.append(self.ResBlock(ch * 2, ch))
            prev = ch

        self.head = nn.Sequential(
            self.ResBlock(channels[0], channels[0]),
            self.RefConv2d(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked

        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i != len(self.encoders) - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = decoder(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)

        return torch.where(valid, y, torch.full_like(y, float("nan")))


