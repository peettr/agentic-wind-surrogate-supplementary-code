import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch):
    groups = min(8, ch)
    while ch % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, ch)


class hard_case_focal_residual_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, bias=False),
                _gn(out_ch),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))

    class _SEBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            hidden = max(1, ch // 8)
            self.fc1 = nn.Conv2d(ch, hidden, 1)
            self.fc2 = nn.Conv2d(hidden, ch, 1)

        def forward(self, x):
            w = F.adaptive_avg_pool2d(x, 1)
            w = F.silu(self.fc1(w))
            w = torch.sigmoid(self.fc2(w))
            return x * w

    class _UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.block = hard_case_focal_residual_unet._ConvBlock(out_ch + skip_ch, out_ch)
            self.attn = hard_case_focal_residual_unet._SEBlock(out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            x = self.block(x)
            return self.attn(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.out_channels = out_channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(2)
        self.bottleneck = nn.Sequential(
            self._ConvBlock(channels[-1], channels[-1]),
            self._SEBlock(channels[-1]),
            self._ConvBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self._UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        nan_mask = torch.isnan(x)
        x_masked = torch.where(nan_mask, torch.zeros_like(x), x)
        invalid = nan_mask.all(dim=1, keepdim=True)

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        skips = skips[:-1][::-1]
        for dec, skip in zip(self.decoders, skips):
            h = dec(h, skip)

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if invalid.any():
            invalid_b = invalid.expand_as(out)
            nan_fill = torch.full_like(out, float("nan"))
            out = torch.where(invalid_b, nan_fill, out)
        return out