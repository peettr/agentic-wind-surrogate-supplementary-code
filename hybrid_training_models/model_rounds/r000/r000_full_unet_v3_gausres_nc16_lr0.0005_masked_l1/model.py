import torch
import torch.nn as nn
import torch.nn.functional as F


class _RefConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad_size = pad
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        if self.pad_size > 0 and (x.shape[-2] <= self.pad_size or x.shape[-1] <= self.pad_size):
            x = F.interpolate(x, size=(max(x.shape[-2], self.pad_size + 1), max(x.shape[-1], self.pad_size + 1)),
                              mode="bilinear", align_corners=False)
        return self.conv(self.pad(x))


class _GausResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _RefConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _RefConv(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

        kernel = torch.tensor(
            [[1.0, 4.0, 6.0, 4.0, 1.0],
             [4.0, 16.0, 24.0, 16.0, 4.0],
             [6.0, 24.0, 36.0, 24.0, 6.0],
             [4.0, 16.0, 24.0, 16.0, 4.0],
             [1.0, 4.0, 6.0, 4.0, 1.0]]
        ) / 256.0
        self.register_buffer("gaus", kernel.view(1, 1, 5, 5))
        self.gpad = nn.ReflectionPad2d(2)

    def forward(self, x):
        residual = self.skip(x)
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))

        if y.shape[-2] > 2 and y.shape[-1] > 2:
            k = self.gaus.expand(y.shape[1], 1, 5, 5)
            smooth = F.conv2d(self.gpad(y), k, groups=y.shape[1])
            y = y + 0.15 * smooth

        if residual.shape[-2:] != y.shape[-2:]:
            residual = F.interpolate(residual, size=y.shape[-2:], mode="bilinear", align_corners=False)

        return F.silu(y + residual)


class unet_v3_gausres(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(_GausResBlock(prev, ch))
            prev = ch

        self.pool = nn.AvgPool2d(2)

        self.bottleneck = nn.Sequential(
            _GausResBlock(channels[-1], channels[-1]),
            _GausResBlock(channels[-1], channels[-1])
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(channels):
            self.up_convs.append(_RefConv(prev, ch, 3, bias=False))
            self.decoders.append(_GausResBlock(ch + ch, ch))
            prev = ch

        self.head = nn.Sequential(
            _RefConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 1, padding=0)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked

        for enc in self.encoders:
            y = enc(y)
            skips.append(y)
            if y.shape[-2] > 1 and y.shape[-1] > 1:
                y = self.pool(y)

        y = self.bottleneck(y)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = F.silu(up(y))
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(y)
        else:
            valid = valid.expand_as(y)

        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y