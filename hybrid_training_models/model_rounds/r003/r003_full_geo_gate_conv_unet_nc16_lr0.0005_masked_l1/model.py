import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _GeoGateBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.gate = nn.Sequential(
            _ReflectConv(out_channels, out_channels, 3, bias=True),
            nn.Sigmoid()
        )
        self.skip = nn.Identity() if in_channels == out_channels else _ReflectConv(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        residual = self.skip(x)
        y = F.silu(self.norm1(self.conv1(x)))
        y = F.silu(self.norm2(self.conv2(y)))
        return F.silu(residual + y * self.gate(y))


class geo_gate_conv_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoder.append(_GeoGateBlock(prev_channels, ch))
            prev_channels = ch

        self.bottleneck = _GeoGateBlock(channels[-1], channels[-1])

        self.up_reduce = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_reduce.append(_ReflectConv(channels[i + 1], channels[i], 1, bias=False))
            self.decoder.append(_GeoGateBlock(channels[i] * 2, channels[i]))

        self.out_head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(),
            _ReflectConv(channels[0], out_channels, 1, bias=True)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i < self.depth - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for reduce, block, skip in zip(self.up_reduce, self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = reduce(y)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        output = self.out_head(y)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(output)

        output = output.clone()
        output[~valid] = float("nan")
        return output