import torch
import torch.nn as nn
import torch.nn.functional as F

def _num_groups(channels):
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups

class _ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            _ReflectionConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            _ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class mean_plus_residual_head_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(_ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = _ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        prev_channels = channels[-1]
        for skip_channels in reversed(channels):
            self.decoders.append(_ConvBlock(prev_channels + skip_channels, skip_channels))
            prev_channels = skip_channels

        final_channels = channels[0]

        self.residual_head = nn.Sequential(
            _ReflectionConv2d(final_channels, final_channels, 3, bias=False),
            nn.GroupNorm(_num_groups(final_channels), final_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(final_channels, out_channels, kernel_size=1, padding=0),
        )

        self.mean_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, kernel_size=1, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(final_channels, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x).any(dim=1, keepdim=True)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked

        for encoder in self.encoders:
            y = encoder(y)
            skips.append(y)
            y = self.down(y)

        y = self.bottleneck(y)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = decoder(y)

        residual = self.residual_head(y)

        valid_float = valid.to(dtype=y.dtype)
        valid_count = valid_float.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)

        residual_mean = (residual * valid_float).sum(dim=(-2, -1), keepdim=True) / valid_count
        residual = residual - residual_mean

        feature_mean = (y * valid_float).sum(dim=(-2, -1), keepdim=True) / valid_count
        mean = self.mean_head(feature_mean)

        output = mean + residual

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_output = valid.expand(-1, output.shape[1], -1, -1)
        output = torch.where(valid_output, output, torch.full_like(output, float("nan")))

        return output