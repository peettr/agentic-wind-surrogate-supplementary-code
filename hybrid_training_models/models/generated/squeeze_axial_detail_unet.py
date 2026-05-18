import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, groups=groups, bias=False)
        self.norm = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(self.pad(x))))


class _SqueezeAxialDetailBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.proj_norm = nn.GroupNorm(min(8, out_channels), out_channels)

        self.dw = _ReflectConv(out_channels, out_channels, 3, groups=out_channels)

        self.h_pad = nn.ReflectionPad2d((3, 3, 0, 0))
        self.h_conv = nn.Conv2d(out_channels, out_channels, (1, 7), padding=0, groups=out_channels, bias=False)

        self.v_pad = nn.ReflectionPad2d((0, 0, 3, 3))
        self.v_conv = nn.Conv2d(out_channels, out_channels, (7, 1), padding=0, groups=out_channels, bias=False)

        self.mix = nn.Conv2d(out_channels, out_channels, 1, padding=0, bias=False)
        self.mix_norm = nn.GroupNorm(min(8, out_channels), out_channels)

        self.detail = nn.Sequential(
            _ReflectConv(out_channels, out_channels, 3, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1, padding=0, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.act(self.proj_norm(self.proj(x)))
        residual = x

        y = self.dw(x)
        y = y + self.h_conv(self.h_pad(y)) + self.v_conv(self.v_pad(y))
        y = self.act(self.mix_norm(self.mix(y)))

        y = y + self.detail(y)
        return y + residual


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(2, 2)
        self.block = _SqueezeAxialDetailBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = _SqueezeAxialDetailBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class squeeze_axial_detail_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = _SqueezeAxialDetailBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            [_Down(channels[i - 1], channels[i]) for i in range(1, depth)]
        )

        self.bottleneck = nn.Sequential(
            _SqueezeAxialDetailBlock(channels[-1], channels[-1]),
            _SqueezeAxialDetailBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList(
            [
                _Up(channels[i], channels[i - 1], channels[i - 1])
                for i in range(depth - 1, 0, -1)
            ]
        )

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for down in self.encoder:
            y = down(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            y = up(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != y.shape[1]:
            valid_out = valid_out.all(dim=1, keepdim=True).expand_as(y)

        y = y.clone()
        y[~valid_out] = float("nan")
        return y