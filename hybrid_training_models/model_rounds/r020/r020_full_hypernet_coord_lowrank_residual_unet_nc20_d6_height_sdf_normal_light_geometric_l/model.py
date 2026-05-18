import torch
import torch.nn as nn
import torch.nn.functional as F


class hypernet_coord_lowrank_residual_unet(nn.Module):
    class ReflectionConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, bias=True):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            groups1 = min(8, out_ch)
            while out_ch % groups1 != 0:
                groups1 -= 1

            self.proj = None
            if in_ch != out_ch:
                self.proj = hypernet_coord_lowrank_residual_unet.ReflectionConv2d(in_ch, out_ch, 1)

            self.conv1 = hypernet_coord_lowrank_residual_unet.ReflectionConv2d(in_ch, out_ch, 3)
            self.norm1 = nn.GroupNorm(groups1, out_ch)
            self.conv2 = hypernet_coord_lowrank_residual_unet.ReflectionConv2d(out_ch, out_ch, 3)
            self.norm2 = nn.GroupNorm(groups1, out_ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            residual = x if self.proj is None else self.proj(x)
            x = self.act(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return self.act(x + residual)

    class CoordLowRankGate(nn.Module):
        def __init__(self, channels, rank=8):
            super().__init__()
            rank = min(rank, channels)
            self.to_rank = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels + 2, rank, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(rank, channels * 2, 1),
            )

        def forward(self, x):
            b, c, h, w = x.shape
            yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
            xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
            pooled = torch.cat([x, xx, yy], dim=1)
            gate, bias = self.to_rank(pooled).chunk(2, dim=1)
            return x * (1.0 + 0.1 * torch.tanh(gate)) + 0.1 * bias

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self.ResBlock(in_channels + 2, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(self.ResBlock(channels[i], channels[i]))

        self.down = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(2),
                self.ResBlock(channels[i], channels[i + 1]),
            )
            for i in range(depth - 1)
        ])

        self.bottleneck = nn.Sequential(
            self.ResBlock(channels[-1], channels[-1]),
            self.CoordLowRankGate(channels[-1]),
            self.ResBlock(channels[-1], channels[-1]),
        )

        self.up_reduce = nn.ModuleList([
            self.ReflectionConv2d(channels[i + 1], channels[i], 1)
            for i in reversed(range(depth - 1))
        ])

        self.decoder = nn.ModuleList([
            self.ResBlock(channels[i] + channels[i], channels[i])
            for i in reversed(range(depth - 1))
        ])

        self.head = nn.Sequential(
            self.ResBlock(channels[0], channels[0]),
            self.ReflectionConv2d(channels[0], out_channels, 3),
        )

    def _coords(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([xx, yy], dim=1)

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        x0 = torch.cat([x_masked, self._coords(x_masked)], dim=1)
        x0 = self.stem(x0)

        skips = []
        h = x0
        h = self.encoder[0](h)
        skips.append(h)

        for i in range(self.depth - 1):
            h = self.down[i](h)
            h = self.encoder[i + 1](h)
            skips.append(h)

        h = self.bottleneck(h)

        for reduce, dec, skip in zip(self.up_reduce, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = reduce(h)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels != valid.shape[1]:
            valid = valid.expand(-1, self.out_channels, -1, -1)

        nan = torch.full_like(out, float("nan"))
        return torch.where(valid, out, nan)


