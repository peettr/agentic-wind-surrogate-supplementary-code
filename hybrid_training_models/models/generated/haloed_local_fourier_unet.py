import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ReflectionConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LocalFourierBlock(nn.Module):
    def __init__(self, channels, modes=24):
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1.0 / max(1, channels)
        self.weight_real = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        self.proj = ReflectionConv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, c, h, w // 2 + 1, dtype=x_ft.dtype, device=x.device)

        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)
        weight = torch.complex(self.weight_real[:, :, :mh, :mw], self.weight_imag[:, :, :mh, :mw])

        out_ft[:, :, :mh, :mw] = torch.einsum("bihw,oihw->bohw", x_ft[:, :, :mh, :mw], weight)

        x_spec = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return self.act(self.norm(self.proj(x) + x_spec + x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = ReflectionConv2d(in_channels, out_channels, 1, bias=False)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class haloed_local_fourier_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            DownBlock(channels[i - 1], channels[i]) for i in range(1, depth)
        )

        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], channels[-1]),
            LocalFourierBlock(channels[-1], modes=24),
            ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList(
            UpBlock(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        )

        self.head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            ReflectionConv2d(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for block in self.encoder:
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = block(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid[:, :1].expand_as(y)
        else:
            valid_out = valid

        y = torch.where(valid_out, y, torch.full_like(y, float("nan")))
        return y


