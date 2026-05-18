import torch
import torch.nn as nn
import torch.nn.functional as F

class unet_v3_gausres(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))

    class GatedGaussianResidual(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.dw = nn.Sequential(
                nn.ReflectionPad2d(2),
                nn.Conv2d(ch, ch, 5, groups=ch, bias=False),
                nn.GroupNorm(min(8, ch), ch),
                nn.SiLU(inplace=True),
            )
            self.pw = nn.Conv2d(ch, ch, 1)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, max(ch // 4, 1), 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(ch // 4, 1), ch, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            y = self.pw(self.dw(x))
            return x + y * self.gate(y)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        chs = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.ConvBlock(in_channels, chs[0])

        self.encoders = nn.ModuleList()
        prev = chs[0]
        for ch in chs[1:]:
            self.encoders.append(nn.Sequential(
                nn.AvgPool2d(2),
                self.ConvBlock(prev, ch),
            ))
            prev = ch

        self.bottleneck = nn.Sequential(
            self.GatedGaussianResidual(chs[-1]),
            self.ConvBlock(chs[-1], chs[-1]),
            self.GatedGaussianResidual(chs[-1]),
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(chs[i + 1], chs[i], 1, bias=False))
            self.decoders.append(self.ConvBlock(chs[i] * 2, chs[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(chs[0], chs[0], 3, bias=False),
            nn.GroupNorm(min(8, chs[0]), chs[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(chs[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for enc in self.encoders:
            y = enc(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(y)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        nan_fill = torch.full_like(y, float("nan"))
        return torch.where(valid.expand_as(y), y, nan_fill)


