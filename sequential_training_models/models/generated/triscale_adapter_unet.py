import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_avg_pool2d(x, kernel_size):
    h, w = x.shape[-2:]
    if h >= kernel_size and w >= kernel_size:
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size)
    return F.adaptive_avg_pool2d(
        x,
        output_size=(max(1, h // kernel_size), max(1, w // kernel_size)),
    )


def _gn(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad_size = pad
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        if self.pad_size == 0:
            return self.conv(x)

        h, w = x.shape[-2:]
        if h > self.pad_size and w > self.pad_size:
            return self.conv(self.pad(x))

        target_h = max(h, self.pad_size + 1)
        target_w = max(w, self.pad_size + 1)
        y = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        y = self.conv(self.pad(y))
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ReflectionConv2d(in_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TriScaleAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduced = max(channels // 4, 8)
        self.local = nn.Sequential(
            ReflectionConv2d(channels, reduced, 3, bias=False),
            _gn(reduced),
            nn.SiLU(inplace=True),
        )
        self.mid = nn.Sequential(
            ReflectionConv2d(channels, reduced, 3, bias=False),
            _gn(reduced),
            nn.SiLU(inplace=True),
        )
        self.global_path = nn.Sequential(
            ReflectionConv2d(channels, reduced, 3, bias=False),
            _gn(reduced),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            ReflectionConv2d(reduced * 3, channels, 3, bias=False),
            _gn(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        y0 = self.local(x)

        y1 = _safe_avg_pool2d(x, kernel_size=2)
        y1 = self.mid(y1)
        y1 = F.interpolate(y1, size=(h, w), mode="bilinear", align_corners=False)

        y2 = _safe_avg_pool2d(x, kernel_size=4)
        y2 = self.global_path(y2)
        y2 = F.interpolate(y2, size=(h, w), mode="bilinear", align_corners=False)

        return x + self.fuse(torch.cat([y0, y1, y2], dim=1))


class triscale_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev_channels = in_channels

        for ch in channels:
            self.encoders.append(ConvBlock(prev_channels, ch))
            self.downs.append(nn.AvgPool2d(kernel_size=2, stride=2))
            prev_channels = ch

        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], channels[-1]),
            TriScaleAdapter(channels[-1]),
            ConvBlock(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for ch in reversed(channels):
            self.up_projs.append(ReflectionConv2d(prev_channels, ch, 3, bias=False))
            self.decoders.append(ConvBlock(ch * 2, ch))
            prev_channels = ch

        self.head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            ReflectionConv2d(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        y = torch.where(valid, x, torch.zeros_like(x))

        skips = []

        for encoder, down in zip(self.encoders, self.downs):
            y = encoder(y)
            skips.append(y)
            if y.shape[-2] >= 2 and y.shape[-1] >= 2:
                y = down(y)

        y = self.bottleneck(y)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = decoder(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.any(dim=1, keepdim=True).expand_as(y)

        return torch.where(valid, y, torch.full_like(y, float("nan")))