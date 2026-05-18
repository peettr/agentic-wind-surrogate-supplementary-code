import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeReflectionPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        self.padding = padding

    def forward(self, x):
        left, right, top, bottom = self.padding
        h, w = x.shape[-2:]

        need_h = max(top, bottom) + 1
        need_w = max(left, right) + 1
        if h < need_h or w < need_w:
            new_h = max(h, need_h)
            new_w = max(w, need_w)
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        return F.pad(x, self.padding, mode="reflect")


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = SafeReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        orig_size = x.shape[-2:]
        y = self.conv(self.pad(x))
        if y.shape[-2:] != orig_size:
            y = F.interpolate(y, size=orig_size, mode="bilinear", align_corners=False)
        return y


class _SizePreservingLocalIntegral(nn.Module):
    def __init__(self, channels, groups):
        super().__init__()
        self.pad = SafeReflectionPad2d(2)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=0)
        self.conv = ReflectionConv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(groups, channels)

    def forward(self, x):
        orig_size = x.shape[-2:]
        y = self.pool(self.pad(x))
        if y.shape[-2:] != orig_size:
            y = F.interpolate(y, size=orig_size, mode="bilinear", align_corners=False)
        y = self.conv(y)
        return self.norm(y)


class LocalIntegralDifferentialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.proj = ReflectionConv2d(in_channels, out_channels, 1, bias=False)
        self.conv1 = ReflectionConv2d(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = ReflectionConv2d(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)

        self.local_integral = _SizePreservingLocalIntegral(out_channels, groups)

        self.differential = nn.Sequential(
            ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(groups, out_channels),
        )

        self.act = nn.SiLU(inplace=False)

    def forward(self, x):
        skip = self.proj(x)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))

        li = self.local_integral(y)
        if li.shape[-2:] != y.shape[-2:]:
            li = F.interpolate(li, size=y.shape[-2:], mode="bilinear", align_corners=False)

        df = self.differential(y)
        if df.shape[-2:] != y.shape[-2:]:
            df = F.interpolate(df, size=y.shape[-2:], mode="bilinear", align_corners=False)

        y = y + li + df
        if skip.shape[-2:] != y.shape[-2:]:
            skip = F.interpolate(skip, size=y.shape[-2:], mode="bilinear", align_corners=False)
        return self.act(y + skip)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = LocalIntegralDifferentialBlock(in_channels, out_channels)

    def forward(self, x):
        skip = self.block(x)
        h, w = skip.shape[-2:]
        if h > 1 and w > 1:
            down = F.avg_pool2d(skip, kernel_size=2, stride=2, ceil_mode=True)
        else:
            down = skip
        return down, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = LocalIntegralDifferentialBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class localized_integral_differential_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7, **kwargs):
        super().__init__()
        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.stem = LocalIntegralDifferentialBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.encoder.append(DownBlock(prev, ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            LocalIntegralDifferentialBlock(channels[-1], channels[-1]),
            LocalIntegralDifferentialBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        rev_channels = list(reversed(channels))
        prev = channels[-1]
        for skip_ch in rev_channels:
            out_ch = skip_ch
            self.decoder.append(UpBlock(prev, skip_ch, out_ch))
            prev = out_ch

        head_groups = min(4, n_c)
        while n_c % head_groups != 0:
            head_groups -= 1

        self.head = nn.Sequential(
            ReflectionConv2d(prev, n_c, 3, bias=False),
            nn.GroupNorm(head_groups, n_c),
            nn.SiLU(inplace=False),
            ReflectionConv2d(n_c, out_channels, 1, bias=True),
        )

    def forward(self, x):
        orig_size = x.shape[-2:]
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        y = self.stem(x_masked)

        skips = []
        for block in self.encoder:
            y, skip = block(y)
            skips.append(skip)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips)):
            y = block(y, skip)

        y = self.head(y)

        if y.shape[-2:] != orig_size:
            y = F.interpolate(y, size=orig_size, mode="bilinear", align_corners=False)

        if valid.shape[1] == y.shape[1]:
            valid_out = valid
        else:
            valid_out = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)
        y = torch.where(valid_out, y, torch.full_like(y, float("nan")))
        return y


