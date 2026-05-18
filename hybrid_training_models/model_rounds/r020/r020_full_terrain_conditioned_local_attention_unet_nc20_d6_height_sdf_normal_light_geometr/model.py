import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for groups in range(min(8, num_channels), 0, -1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = _gn(out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels, 3, bias=False)
        self.norm2 = _gn(out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.silu(y + self.skip(x))


class _TerrainConditionedLocalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 4)
        self.local = nn.Sequential(
            _ReflectConv(channels, channels, 3, groups=channels, bias=False),
            nn.Conv2d(channels, hidden, 1, padding=0, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.terrain_gate = nn.Sequential(
            _ReflectConv(1, hidden, 3, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x, terrain):
        terrain = F.interpolate(terrain, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * (1.0 + self.local(x) * self.terrain_gate(terrain))


class terrain_conditioned_local_attention_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_adapter = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.terrain_adapter = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

        self.encoders = nn.ModuleList()
        self.attn = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(nn.Sequential(_ResidualBlock(prev, ch), _ResidualBlock(ch, ch)))
            self.attn.append(_TerrainConditionedLocalAttention(ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            _ResidualBlock(channels[-1], channels[-1]),
            _ResidualBlock(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0, bias=False))
            self.decoders.append(nn.Sequential(_ResidualBlock(channels[i] * 2, channels[i]), _ResidualBlock(channels[i], channels[i])))

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 1, padding=0, bias=True),
        )

    def forward(self, x):
        nan_mask = torch.isnan(x)
        x_masked = torch.where(nan_mask, torch.zeros_like(x), x)

        x_in = self.input_adapter(x_masked)
        terrain = self.terrain_adapter(x_masked)

        skips = []
        y = x_in

        for i, (encoder, attn) in enumerate(zip(self.encoders, self.attn)):
            y = attn(encoder(y), terrain)
            skips.append(y)
            if i < self.depth - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = decoder(torch.cat([y, skip], dim=1))

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if y.shape[1] == 1:
            valid = (~nan_mask).all(dim=1, keepdim=True)
        else:
            if nan_mask.shape[1] == y.shape[1]:
                valid = ~nan_mask
            else:
                valid = (~nan_mask).all(dim=1, keepdim=True).expand_as(y)

        y = y.masked_fill(~valid, float("nan"))
        return y


