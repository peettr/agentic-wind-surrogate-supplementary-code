import torch
import torch.nn as nn
import torch.nn.functional as F


class lowrank_spatial_operator_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class LowRankSpatialOperator(nn.Module):
        def __init__(self, channels, rank=16):
            super().__init__()
            rank = min(rank, channels)
            self.to_rank = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
            self.mix = nn.Sequential(
                nn.AdaptiveAvgPool2d((10, 10)),
                nn.ReflectionPad2d(1),
                nn.Conv2d(rank, rank, kernel_size=3, groups=rank, bias=False),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(rank, rank, kernel_size=3, groups=rank, bias=False),
                nn.SiLU(inplace=True),
            )
            self.from_rank = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
            self.scale = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            h, w = x.shape[-2:]
            z = self.to_rank(x)
            z = self.mix(z)
            z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
            return x + self.scale * self.from_rank(z)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            self.ConvBlock(bottleneck_ch, bottleneck_ch),
            self.LowRankSpatialOperator(bottleneck_ch, rank=max(4, n_c)),
            self.ConvBlock(bottleneck_ch, bottleneck_ch),
        )

        self.decoders = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        cur_ch = bottleneck_ch
        for skip_ch in reversed(channels):
            self.up_projs.append(nn.Conv2d(cur_ch, skip_ch, kernel_size=1, bias=False))
            self.decoders.append(self.ConvBlock(skip_ch * 2, skip_ch))
            cur_ch = skip_ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        output_valid = valid
        if output_valid.shape[1] != out.shape[1]:
            output_valid = output_valid.all(dim=1, keepdim=True)

        nan_fill = torch.full_like(out, float("nan"))
        out = torch.where(output_valid, out, nan_fill)
        return out