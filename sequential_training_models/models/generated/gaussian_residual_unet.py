import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.conv1 = ReflectionConv2d(in_channels, out_channels, 3)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = ReflectionConv2d(out_channels, out_channels, 3)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else ReflectionConv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super().__init__()
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        weight = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.register_buffer("weight", weight)
        self.channels = channels

    def forward(self, x):
        return F.conv2d(self.pad(x), self.weight, groups=self.channels)


class gaussian_residual_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.input_proj = ReflectionConv2d(in_channels, channels[0], 3)
        self.input_smooth = GaussianSmoothing(channels[0], kernel_size=5, sigma=1.0)

        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()

        prev_channels = channels[0]
        for ch in channels:
            self.encoder.append(ResidualBlock(prev_channels, ch))
            self.down.append(nn.AvgPool2d(kernel_size=2, stride=2))
            prev_channels = ch

        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1]),
            ResidualBlock(channels[-1], channels[-1]),
        )

        self.up_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()

        current_channels = channels[-1]
        for skip_channels in reversed(channels):
            self.up_proj.append(ReflectionConv2d(current_channels, skip_channels, 1))
            self.decoder.append(ResidualBlock(skip_channels * 2, skip_channels))
            current_channels = skip_channels

        self.output_head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3),
            nn.SiLU(),
            ReflectionConv2d(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        h0 = self.input_proj(x_masked)
        h = h0 + self.input_smooth(h0)

        skips = []
        for block, down in zip(self.encoder, self.down):
            h = block(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for proj, block, skip in zip(self.up_proj, self.decoder, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = proj(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        residual = self.output_head(h)
        out = x_masked + residual
        out = torch.where(valid, out, torch.full_like(out, float("nan")))
        return out