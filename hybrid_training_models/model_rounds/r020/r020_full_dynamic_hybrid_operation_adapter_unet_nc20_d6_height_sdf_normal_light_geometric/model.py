import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class HybridAdapterBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ReflectionConv2d(in_channels, out_channels, 3)
        self.norm1 = _gn(out_channels)
        self.conv2 = ReflectionConv2d(out_channels, out_channels, 3)
        self.norm2 = _gn(out_channels)

        hidden_channels = max(1, out_channels // 4)
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.Sigmoid(),
        )

        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, padding=0)

    def forward(self, x):
        residual = self.skip(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.silu(y, inplace=True)
        y = self.conv2(y)
        y = self.norm2(y)
        y = y * self.adapter(y)
        return F.silu(y + residual, inplace=True)


class dynamic_hybrid_operation_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoder.append(HybridAdapterBlock(prev_channels, ch))
            prev_channels = ch

        self.bottleneck = HybridAdapterBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(HybridAdapterBlock(channels[i + 1], channels[i]))
            self.fuse.append(HybridAdapterBlock(channels[i] * 2, channels[i]))

        self.out_head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i != self.depth - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for dec_block, fuse_block, skip in zip(self.decoder, self.fuse, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = dec_block(y)
            y = torch.cat([y, skip], dim=1)
            y = fuse_block(y)

        y = self.out_head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels == self.in_channels:
            y = torch.where(valid, y, torch.full_like(y, float("nan")))
        else:
            out_valid = valid.any(dim=1, keepdim=True).expand(-1, self.out_channels, -1, -1)
            y = torch.where(out_valid, y, torch.full_like(y, float("nan")))

        return y