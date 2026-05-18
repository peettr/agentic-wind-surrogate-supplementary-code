import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for num_groups in range(min(8, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Identity()
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.net(x) + self.proj(x)


class wavelet_refine_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch

        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1, bias=False))
            self.decoders.append(ConvBlock(channels[i] * 2, channels[i]))

        self.refine = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0] + in_channels, channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        h = torch.cat([h, x_masked], dim=1)
        h = self.refine(h)
        out = self.out_conv(h)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            out_valid = out_valid.any(dim=1, keepdim=True)
        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out