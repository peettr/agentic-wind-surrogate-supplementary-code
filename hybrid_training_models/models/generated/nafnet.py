import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectionConv2d(nn.Module):
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

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, channels, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, dw_channels, kernel_size=1, padding=0)
        self.dw = ReflectionConv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            stride=1,
            groups=dw_channels,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, padding=0),
        )
        self.pw2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, padding=0)

        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn1 = nn.Conv2d(channels, ffn_channels, kernel_size=1, padding=0)
        self.ffn2 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, padding=0)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.sg(y)
        y = y * self.sca(y)
        y = self.pw2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.sg(y)
        y = self.ffn2(y)
        return x + y * self.gamma

class nafnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=4):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.intro = ReflectionConv2d(in_channels, channels[0], kernel_size=3)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, ch in enumerate(channels):
            self.encoders.append(nn.Sequential(NAFBlock(ch), NAFBlock(ch)))
            if i < depth - 1:
                self.downs.append(nn.Conv2d(ch, channels[i + 1], kernel_size=2, stride=2, padding=0))

        self.bottleneck = nn.Sequential(NAFBlock(channels[-1]), NAFBlock(channels[-1]))

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels[i + 1], channels[i] * 4, kernel_size=1, padding=0),
                    nn.PixelShuffle(2),
                )
            )
            self.decoders.append(nn.Sequential(NAFBlock(channels[i]), NAFBlock(channels[i])))

        self.ending = ReflectionConv2d(channels[0], out_channels, kernel_size=3)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        h, w = x_masked.shape[-2:]
        y = self.intro(x_masked)

        skips = []
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            if i < len(self.downs):
                skips.append(y)
                y = self.downs[i](y)

        y = self.bottleneck(y)

        for up, decoder in zip(self.ups, self.decoders):
            skip = skips.pop()
            y = up(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = decoder(y + skip)

        y = self.ending(y)
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != y.shape[1]:
            valid_out = valid_out[:, :1].expand_as(y)

        return torch.where(valid_out, y, torch.full_like(y, float("nan")))