import torch
import torch.nn as nn
import torch.nn.functional as F

class _ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(self.pad(x))))

class _MSCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = _ConvBNAct(channels, channels, 3, groups=channels)
        self.k5 = _ConvBNAct(channels, channels, 5, groups=channels)
        self.k7 = _ConvBNAct(channels, channels, 7, groups=channels)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        a = torch.cat((self.local(x), self.k5(x), self.k7(x)), dim=1)
        return x * self.mix(a)

class _Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)
        self.conv1 = _ConvBNAct(in_ch, out_ch, 3)
        self.attn = _MSCA(out_ch)
        self.conv2 = _ConvBNAct(out_ch, out_ch, 3)

    def forward(self, x):
        r = self.proj(x)
        x = self.conv1(x)
        x = self.attn(x)
        x = self.conv2(x)
        return x + r

class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = _ConvBNAct(in_ch, out_ch, 3, stride=2)
        self.block = _Block(out_ch, out_ch)

    def forward(self, x):
        return self.block(self.down(x))

class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)
        self.block = _Block(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat((x, skip), dim=1)
        return self.block(x)

class msca_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = _Block(in_channels, channels[0])
        self.encoder = nn.ModuleList([
            _Down(channels[i - 1], channels[i]) for i in range(1, depth)
        ])

        self.bottleneck = nn.Sequential(
            _Block(channels[-1], channels[-1]),
            _Block(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList([
            _Up(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        ])

        self.head = nn.Sequential(
            _ConvBNAct(channels[0], channels[0], 3),
            nn.Conv2d(channels[0], out_channels, 1, padding=0, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        x = self.stem(x_masked)
        skips.append(x)

        for down in self.encoder:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            x = up(x, skip)

        x = self.head(x)

        if x.shape[-2:] != x_masked.shape[-2:]:
            x = F.interpolate(x, size=x_masked.shape[-2:], mode="bilinear", align_corners=False)

        return torch.where(valid, x, torch.full_like(x, float("nan")))