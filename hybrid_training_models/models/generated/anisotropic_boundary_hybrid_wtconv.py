import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class anisotropic_boundary_hybrid_wtconv(nn.Module):
    class RefConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, bias=True):
            super().__init__()
            if isinstance(kernel_size, tuple):
                pad = (kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] // 2, kernel_size[0] // 2)
            else:
                pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, groups=groups, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                anisotropic_boundary_hybrid_wtconv.RefConv2d(in_ch, out_ch, 3),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                anisotropic_boundary_hybrid_wtconv.RefConv2d(out_ch, out_ch, (1, 7), groups=1),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                anisotropic_boundary_hybrid_wtconv.RefConv2d(out_ch, out_ch, (7, 1), groups=1),
                _gn(out_ch),
            )
            self.skip = nn.Identity() if in_ch == out_ch else anisotropic_boundary_hybrid_wtconv.RefConv2d(in_ch, out_ch, 1)

        def forward(self, x):
            return F.silu(self.net(x) + self.skip(x), inplace=True)

    class DownBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.down = anisotropic_boundary_hybrid_wtconv.RefConv2d(in_ch, out_ch, 3, stride=2)
            self.block = anisotropic_boundary_hybrid_wtconv.ConvBlock(out_ch, out_ch)

        def forward(self, x):
            return self.block(self.down(x))

    class UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.fuse = anisotropic_boundary_hybrid_wtconv.ConvBlock(in_ch + skip_ch, out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.fuse(torch.cat([x, skip], dim=1))

    class Bottleneck(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.local = anisotropic_boundary_hybrid_wtconv.ConvBlock(ch, ch)
            self.mix = nn.Sequential(
                anisotropic_boundary_hybrid_wtconv.RefConv2d(ch, ch, 3, groups=ch),
                _gn(ch),
                nn.SiLU(inplace=True),
                anisotropic_boundary_hybrid_wtconv.RefConv2d(ch, ch, 1),
            )

        def forward(self, x):
            return self.local(x) + self.mix(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        max_ch = n_c * 8
        channels = [min(n_c * (2 ** i), max_ch) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stem = self.ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList([
            self.DownBlock(channels[i], channels[i + 1]) for i in range(depth - 1)
        ])
        self.bottleneck = self.Bottleneck(channels[-1])
        self.decoder = nn.ModuleList([
            self.UpBlock(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        ])
        self.head = nn.Sequential(
            self.ConvBlock(channels[0], channels[0]),
            self.RefConv2d(channels[0], out_channels, 1)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        if x.shape[1] != self.in_channels:
            if x.shape[1] > self.in_channels:
                x_in = x[:, :self.in_channels]
            else:
                reps = [1] * x.dim()
                reps[1] = self.in_channels - x.shape[1] + 1
                pad = x[:, :1].repeat(*reps)[:, : self.in_channels - x.shape[1]]
                x_in = torch.cat([x, pad], dim=1)
        else:
            x_in = x

        valid = ~torch.isnan(x_in)
        valid_any = valid.all(dim=1, keepdim=True)
        x_masked = torch.where(valid, x_in, torch.zeros_like(x_in))

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

        mask = valid_any.expand_as(y)
        return torch.where(mask, y, torch.full_like(y, float("nan")))