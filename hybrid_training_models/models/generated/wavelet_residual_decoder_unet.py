import torch
import torch.nn as nn
import torch.nn.functional as F


class wavelet_residual_decoder_unet(nn.Module):
    class ReflectConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            if x.shape[-2] <= self.pad.padding[0] or x.shape[-1] <= self.pad.padding[1]:
                x = F.interpolate(x, size=(max(x.shape[-2], 2), max(x.shape[-1], 2)), mode="nearest")
                x = self.conv(self.pad(x))
                return F.interpolate(x, size=(1, 1), mode="nearest")
            return self.conv(self.pad(x))

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = wavelet_residual_decoder_unet.ReflectConv2d(in_channels, out_channels, 3, bias=False)
            self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
            self.conv2 = wavelet_residual_decoder_unet.ReflectConv2d(out_channels, out_channels, 3, bias=False)
            self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
            )

        def forward(self, x):
            residual = self.skip(x)
            x = F.silu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return F.silu(x + residual)

    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.pool = nn.AvgPool2d(2, 2)
            self.block = wavelet_residual_decoder_unet.ResidualBlock(in_channels, out_channels)

        def forward(self, x):
            return self.block(self.pool(x))

    class UpBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.reduce = wavelet_residual_decoder_unet.ReflectConv2d(in_channels, out_channels, 3, bias=False)
            self.block = wavelet_residual_decoder_unet.ResidualBlock(out_channels + skip_channels, out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = F.silu(self.reduce(x))
            x = torch.cat([x, skip], dim=1)
            return self.block(x)

    class WaveletBottleneck(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.low = wavelet_residual_decoder_unet.ResidualBlock(channels, channels)
            self.high = wavelet_residual_decoder_unet.ResidualBlock(channels, channels)
            self.mix = wavelet_residual_decoder_unet.ReflectConv2d(channels * 2, channels, 3, bias=False)
            self.norm = nn.GroupNorm(min(8, channels), channels)

        def forward(self, x):
            if x.shape[-2] < 2 or x.shape[-1] < 2:
                return self.high(x)

            low = F.avg_pool2d(x, 2, 2)
            low = self.low(low)
            low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
            high = self.high(x - low)
            return F.silu(self.norm(self.mix(torch.cat([low, high], dim=1))) + x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.stem = self.ResidualBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            self.DownBlock(channels[i], channels[i + 1]) for i in range(depth - 1)
        )
        self.bottom_down = nn.AvgPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            self.ResidualBlock(channels[-1], channels[-1]),
            self.WaveletBottleneck(channels[-1]),
            self.ResidualBlock(channels[-1], channels[-1]),
        )
        self.decoder = nn.ModuleList(
            self.UpBlock(
                channels[i + 1] if i < depth - 1 else channels[-1],
                channels[i],
                channels[i],
            )
            for i in reversed(range(depth))
        )
        self.head = nn.Sequential(
            self.ResidualBlock(channels[0], channels[0]),
            self.ReflectConv2d(channels[0], out_channels, 3, bias=True),
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

        y = self.bottom_down(y)
        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips)):
            y = block(y, skip)

        output = self.head(y)

        if valid.shape[1] != output.shape[1]:
            valid = valid[:, :1].expand_as(output)
        else:
            valid = valid.expand_as(output)

        output = output.clone()
        output[~valid] = float("nan")
        return output


