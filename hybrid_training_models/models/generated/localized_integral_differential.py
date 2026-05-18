import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class localized_integral_differential(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
                _gn(out_channels),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
                _gn(out_channels),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class IntegralDifferentialBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.local = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3, groups=channels, bias=False),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                _gn(channels),
                nn.GELU(),
            )
            self.integral = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(channels, channels, kernel_size=7, groups=channels, bias=False),
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                _gn(channels),
                nn.GELU(),
            )
            self.mix = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                _gn(channels),
                nn.GELU(),
            )

        def forward(self, x):
            y = torch.cat([self.local(x), self.integral(x)], dim=1)
            return x + self.mix(y)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        self.out_channels = out_channels
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            self.IntegralDifferentialBlock(bottleneck_channels),
            self.IntegralDifferentialBlock(bottleneck_channels),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1, bias=False))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, bias=False),
            _gn(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            x_masked = torch.where(nan_mask, torch.zeros_like(x), x)
        else:
            x_masked = x

        spatial_nan = nan_mask.any(dim=1, keepdim=True)

        skips = []
        y = x_masked
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i != len(self.encoders) - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = decoder(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if spatial_nan.any():
            mask = spatial_nan.expand(-1, y.shape[1], -1, -1)
            y = torch.where(mask, torch.full_like(y, float("nan")), y)
        return y