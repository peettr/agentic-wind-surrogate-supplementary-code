import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for groups in range(min(8, num_channels), 0, -1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class _ReflectionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if isinstance(kernel_size, int):
            pad = kernel_size // 2
            padding = (pad, pad, pad, pad)
        else:
            kh, kw = kernel_size
            padding = (kw // 2, kw // 2, kh // 2, kh // 2)
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False)

    def forward(self, x):
        return self.conv(self.pad(x))


class _HybridBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        )
        self.conv1 = _ReflectionConv(in_channels, out_channels, 3)
        self.norm1 = _gn(out_channels)
        self.conv_h = _ReflectionConv(out_channels, out_channels, (1, 5))
        self.conv_v = _ReflectionConv(out_channels, out_channels, (5, 1))
        self.mix = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, padding=0, bias=False)
        self.norm2 = _gn(out_channels)
        self.conv2 = _ReflectionConv(out_channels, out_channels, 3)
        self.norm3 = _gn(out_channels)

    def forward(self, x):
        residual = self.proj(x)
        x = F.silu(self.norm1(self.conv1(x)), inplace=True)
        h = self.conv_h(x)
        v = self.conv_v(x)
        x = F.silu(self.norm2(self.mix(torch.cat([h, v], dim=1))), inplace=True)
        x = self.norm3(self.conv2(x))
        return F.silu(x + residual, inplace=True)


class anisotropic_boundary_hybrid_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(_HybridBlock(prev, ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            _HybridBlock(channels[-1], channels[-1]),
            _HybridBlock(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1, padding=0, bias=False))
            self.decoders.append(_HybridBlock(channels[i] * 2, channels[i]))

        self.out_head = nn.Sequential(
            _HybridBlock(channels[0], channels[0]),
            _ReflectionConv(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i != self.depth - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = decoder(torch.cat([y, skip], dim=1))

        y = self.out_head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)
        else:
            valid_out = valid
        y = torch.where(valid_out, y, torch.full_like(y, float("nan")))
        return y