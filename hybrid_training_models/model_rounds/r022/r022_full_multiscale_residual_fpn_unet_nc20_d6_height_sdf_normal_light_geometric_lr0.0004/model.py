import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels, max_groups=8):
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _ReflectConv(channels, channels, 3, bias=False)
        self.norm1 = _gn(channels)
        self.conv2 = _ReflectConv(channels, channels, 3, bias=False)
        self.norm2 = _gn(channels)

    def forward(self, x):
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)


class _Stage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = _ReflectConv(in_channels, out_channels, 3, bias=False)
        self.norm = _gn(out_channels)
        self.res1 = _ResidualBlock(out_channels)
        self.res2 = _ResidualBlock(out_channels)

    def forward(self, x):
        x = F.gelu(self.norm(self.proj(x)))
        x = self.res1(x)
        x = self.res2(x)
        return x


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = _ReflectConv(in_channels, out_channels, 3, stride=2, bias=False)
        self.norm = _gn(out_channels)
        self.res = _ResidualBlock(out_channels)

    def forward(self, x):
        x = F.gelu(self.norm(self.down(x)))
        return self.res(x)


class _Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.skip_proj = _ReflectConv(skip_channels, out_channels, 1, bias=False)
        self.fuse = _Stage(in_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.skip_proj(skip)
        return self.fuse(torch.cat([x, skip], dim=1))


class multiscale_residual_fpn_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.channels = channels

        self.stem = _Stage(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            [_Down(channels[i], channels[i + 1]) for i in range(depth - 1)]
        )

        self.bottleneck = nn.Sequential(
            _ResidualBlock(channels[-1]),
            _ResidualBlock(channels[-1]),
        )

        self.lateral = nn.ModuleList(
            [_ReflectConv(ch, channels[-1], 1, bias=False) for ch in channels]
        )

        self.decoder = nn.ModuleList()
        cur_channels = channels[-1]
        for i in range(depth - 2, -1, -1):
            self.decoder.append(_Up(cur_channels, channels[-1], channels[i]))
            cur_channels = channels[i]

        self.head = nn.Sequential(
            _ResidualBlock(channels[0]),
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.GELU(),
            _ReflectConv(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x).any(dim=1, keepdim=True)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for down in self.encoder:
            y = down(y)
            skips.append(y)

        y = self.bottleneck(y)

        fpn = self.lateral[-1](skips[-1])
        fpn_skips = [None] * self.depth
        fpn_skips[-1] = fpn

        for i in range(self.depth - 2, -1, -1):
            fpn = F.interpolate(fpn, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            fpn = fpn + self.lateral[i](skips[i])
            fpn_skips[i] = fpn

        for i, up in enumerate(self.decoder):
            skip = fpn_skips[self.depth - 2 - i]
            y = up(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.expand(-1, y.shape[1], -1, -1)

        return torch.where(valid, y, torch.full_like(y, float("nan")))