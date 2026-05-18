import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels):
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class boundary_assimilated_spectral_residual_unet(nn.Module):
    class ReflectionConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
            super().__init__()
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = boundary_assimilated_spectral_residual_unet.ReflectionConv2d(in_channels, out_channels, 3)
            self.norm1 = _gn(out_channels)
            self.conv2 = boundary_assimilated_spectral_residual_unet.ReflectionConv2d(out_channels, out_channels, 3)
            self.norm2 = _gn(out_channels)
            self.skip = nn.Identity()
            if in_channels != out_channels:
                self.skip = boundary_assimilated_spectral_residual_unet.ReflectionConv2d(in_channels, out_channels, 1)

        def forward(self, x):
            residual = self.skip(x)
            x = F.gelu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return F.gelu(x + residual)

    class SpectralResidualBlock(nn.Module):
        def __init__(self, channels, modes=24):
            super().__init__()
            self.channels = channels
            self.modes = modes
            scale = 1.0 / max(1, channels)
            self.weight_real = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
            self.weight_imag = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
            self.local = boundary_assimilated_spectral_residual_unet.ResidualBlock(channels, channels)

        def forward(self, x):
            b, c, h, w = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros(b, c, h, w // 2 + 1, dtype=x_ft.dtype, device=x.device)

            mh = min(self.modes, h)
            mw = min(self.modes, w // 2 + 1)
            weight = torch.complex(
                self.weight_real[:, :, :mh, :mw],
                self.weight_imag[:, :, :mh, :mw],
            )
            out_ft[:, :, :mh, :mw] = torch.einsum("bihw,iohw->bohw", x_ft[:, :, :mh, :mw], weight)

            spectral = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
            return self.local(x) + spectral

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.boundary_embed = self.ReflectionConv2d(in_channels * 2, channels[0], 3)

        self.encoder = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.encoder.append(self.ResidualBlock(prev, ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            self.SpectralResidualBlock(channels[-1], modes=16),
            self.ResidualBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self.ResidualBlock(channels[i + 1], channels[i]))
            self.fuse.append(self.ResidualBlock(channels[i] + channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ResidualBlock(channels[0], channels[0]),
            self.ReflectionConv2d(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))
        valid_float = valid.to(dtype=x_masked.dtype)

        h, w = x_masked.shape[-2:]
        y = torch.cat([x_masked, valid_float], dim=1)
        y = self.boundary_embed(y)

        skips = []
        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i != len(self.encoder) - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for block, fuse, skip in zip(self.decoder, self.fuse, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(y)
            y = fuse(torch.cat([y, skip], dim=1))

        y = self.head(y)
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

        if y.shape[1] == valid.shape[1]:
            y = y.masked_fill(~valid, float("nan"))
        else:
            y_valid = valid.any(dim=1, keepdim=True).expand_as(y)
            y = y.masked_fill(~y_valid, float("nan"))

        return y


