import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.GroupNorm(self._groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
            nn.GroupNorm(self._groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _groups(channels):
        for g in (8, 4, 2):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x):
        return self.net(x)


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, groups=channels, bias=False),
            nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
            nn.GroupNorm(_ConvBlock._groups(channels * 2), channels * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_ConvBlock._groups(channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.block = _ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = _ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class mambairv2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = _ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            [_Down(channels[i - 1], channels[i]) for i in range(1, depth)]
        )

        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            _ResidualBlock(bottleneck_channels),
            _ResidualBlock(bottleneck_channels),
            _ResidualBlock(bottleneck_channels),
        )

        self.decoder = nn.ModuleList(
            [
                _Up(channels[i], channels[i - 1], channels[i - 1])
                for i in range(depth - 1, 0, -1)
            ]
        )

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for down in self.encoder:
            y = down(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            y = up(y, skip)

        output = self.head(y)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != output.shape[1]:
            valid_out = valid_out[:, :1].expand_as(output)

        output = output.clone()
        output[~valid_out] = float("nan")
        return output