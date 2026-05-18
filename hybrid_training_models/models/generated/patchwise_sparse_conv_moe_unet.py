import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels, max_groups=8):
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class SparseMoEBlock(nn.Module):
    def __init__(self, channels, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        hidden = max(channels // 4, 8)
        groups = _num_groups(channels)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, num_experts, 1),
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                ReflectionConv2d(channels, channels, 3, bias=False),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
                ReflectionConv2d(channels, channels, 3, bias=False),
                nn.GroupNorm(groups, channels),
            )
            for _ in range(num_experts)
        ])

        self.act = nn.GELU()

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=1)
        out = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            out = out + expert(x) * weights[:, i:i + 1]

        return self.act(out + x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = _num_groups(out_channels)

        self.block = nn.Sequential(
            ReflectionConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.moe = SparseMoEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.moe(x)
        pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x, pooled


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        self.moe = SparseMoEBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.moe(x)


class patchwise_sparse_conv_moe_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels

        for ch in channels[:-1]:
            self.encoders.append(DownBlock(prev_channels, ch))
            prev_channels = ch

        self.bottleneck = nn.Sequential(
            ConvBlock(prev_channels, channels[-1]),
            SparseMoEBlock(channels[-1]),
            SparseMoEBlock(channels[-1]),
        )

        self.decoders = nn.ModuleList()
        decoder_in = channels[-1]

        for skip_channels in reversed(channels[:-1]):
            self.decoders.append(UpBlock(decoder_in, skip_channels, skip_channels))
            decoder_in = skip_channels

        self.out_head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(_num_groups(channels[0]), channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for encoder in self.encoders:
            skip, h = encoder(h)
            skips.append(skip)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            h = decoder(h, skip)

        out = self.out_head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid = valid.all(dim=1, keepdim=True)

        out = torch.where(valid, out, torch.full_like(out, float("nan")))
        return out


