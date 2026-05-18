import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

        self.conv1 = _ReflectConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

    def forward(self, x):
        residual = x if self.proj is None else self.proj(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class _DepthwiseMix(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = _ReflectConv(channels, channels, 3, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1, padding=0)
        self.pw2 = nn.Conv2d(channels * 2, channels, 1, padding=0)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.pw2(F.gelu(self.pw1(x)))
        x = self.norm(x)
        return F.gelu(x + residual)


class _LiteBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local1 = _DepthwiseMix(channels)
        self.local2 = _DepthwiseMix(channels)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 1), 1, padding=0),
            nn.GELU(),
            nn.Conv2d(max(channels // 4, 1), channels, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.local1(x)
        x = self.local2(x)
        return x * self.gate(x)


class transolver_lite(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=4):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.channels = channels

        self.stem = _Block(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(nn.AvgPool2d(2, 2))
            self.encoders.append(_Block(channels[i - 1], channels[i]))

        self.bottleneck = _LiteBottleneck(channels[-1])

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0, bias=False))
            self.decoders.append(_Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for down, encoder in zip(self.downs, self.encoders):
            y = down(y)
            y = encoder(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = decoder(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)
        else:
            valid_out = valid

        nan_fill = torch.full_like(y, float("nan"))
        return torch.where(valid_out, y, nan_fill)