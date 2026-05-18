import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = _ReflectConv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.conv1 = _ReflectConv2d(in_channels, out_channels, kernel_size=3)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)

        self.dw_h = _ReflectConv2d(out_channels, out_channels, kernel_size=3, groups=out_channels)
        self.dw_v = _ReflectConv2d(out_channels, out_channels, kernel_size=3, groups=out_channels)
        self.mix = _ReflectConv2d(out_channels, out_channels, kernel_size=1)

        self.conv2 = _ReflectConv2d(out_channels, out_channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

    def forward(self, x):
        residual = self.proj(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)

        h = torch.cumsum(x, dim=3) + torch.flip(torch.cumsum(torch.flip(x, dims=[3]), dim=3), dims=[3])
        v = torch.cumsum(x, dim=2) + torch.flip(torch.cumsum(torch.flip(x, dims=[2]), dim=2), dims=[2])
        h = h / max(1, x.shape[3])
        v = v / max(1, x.shape[2])

        x = x + self.dw_h(h) + self.dw_v(v)
        x = self.mix(x)
        x = F.gelu(x)

        x = self.conv2(x)
        x = self.norm2(x)

        return F.gelu(x + residual)


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = _ReflectConv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.block = _Block(out_channels, out_channels)

    def forward(self, x):
        return self.block(self.down(x))


class _Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = _ReflectConv2d(in_channels, out_channels, kernel_size=1)
        self.block = _Block(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class noncausal_neighbor_ssm_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.channels = channels

        self.stem = _Block(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(_Down(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            _Block(channels[-1], channels[-1]),
            _Block(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(_Up(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            _Block(channels[0], channels[0]),
            _ReflectConv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for down in self.encoder:
            y = down(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            y = up(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid[:, :1].expand(-1, y.shape[1], -1, -1)
        else:
            valid_out = valid

        y = torch.where(valid_out, y, torch.full_like(y, float("nan")))
        return y