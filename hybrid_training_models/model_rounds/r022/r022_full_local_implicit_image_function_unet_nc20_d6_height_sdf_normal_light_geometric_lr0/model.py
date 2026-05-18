import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    groups = min(8, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ReflectionConv2d(in_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class local_implicit_image_function_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        depth = max(1, int(depth))
        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoder.append(ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.up_blocks = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for skip_ch in reversed(channels[:-1]):
            self.up_blocks.append(ReflectionConv2d(prev_channels, skip_ch, 3, bias=False))
            self.decoder.append(ConvBlock(skip_ch * 2, skip_ch))
            prev_channels = skip_ch

        self.head = nn.Sequential(
            ReflectionConv2d(prev_channels, prev_channels, 3, bias=False),
            _gn(prev_channels),
            nn.SiLU(inplace=True),
            ReflectionConv2d(prev_channels, out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i < len(self.encoder) - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_blocks, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid_mask = valid.any(dim=1, keepdim=True).expand(-1, output.shape[1], -1, -1)
        else:
            valid_mask = valid
        output = torch.where(valid_mask, output, torch.full_like(output, float("nan")))
        return output