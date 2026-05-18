import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    groups = min(8, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class cross_shape_meta_adapter_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
            )
            self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.net(x) + self.proj(x)

    class CrossShapeBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.h = nn.Sequential(
                nn.ReflectionPad2d((2, 2, 0, 0)),
                nn.Conv2d(ch, ch, kernel_size=(1, 5), padding=0, groups=ch, bias=False),
            )
            self.v = nn.Sequential(
                nn.ReflectionPad2d((0, 0, 2, 2)),
                nn.Conv2d(ch, ch, kernel_size=(5, 1), padding=0, groups=ch, bias=False),
            )
            self.mix = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
            self.norm = _gn(ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            y = self.h(x) + self.v(x)
            return self.act(self.norm(self.mix(y)) + x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.CrossShapeBlock(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        rev_channels = list(reversed(channels))
        in_ch = rev_channels[0]
        for skip_ch in rev_channels[1:]:
            self.decoders.append(self.ConvBlock(in_ch + skip_ch, skip_ch))
            in_ch = skip_ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i < len(self.encoders) - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = decoder(y)

        y = self.head(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if y.shape[1] == valid.shape[1]:
            y = torch.where(valid, y, torch.full_like(y, float("nan")))
        else:
            out_valid = valid[:, :1].expand(-1, y.shape[1], -1, -1)
            y = torch.where(out_valid, y, torch.full_like(y, float("nan")))

        return y


