import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class multiwavelet_multigrid_detail_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                multiwavelet_multigrid_detail_unet.ReflectConv(in_ch, out_ch),
                multiwavelet_multigrid_detail_unet.ReflectConv(out_ch, out_ch),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class DetailGate(nn.Module):
        def __init__(self, low_ch, skip_ch):
            super().__init__()
            self.low_proj = nn.Identity() if low_ch == skip_ch else nn.Conv2d(low_ch, skip_ch, kernel_size=1, bias=False)
            self.mix = nn.Sequential(
                nn.Conv2d(skip_ch * 2, skip_ch, kernel_size=1, bias=False),
                _gn(skip_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(skip_ch, skip_ch, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, low, skip):
            low_up = F.interpolate(low, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            low_up = self.low_proj(low_up)
            detail = skip - low_up
            return skip + self.mix(torch.cat([skip, detail], dim=1)) * detail

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.Block(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(self.Block(channels[i - 1], channels[i]))

        self.bottleneck = self.Block(channels[-1], channels[-1])

        self.detail_gates = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.detail_gates.append(self.DetailGate(channels[i + 1], channels[i]))
            self.decoder.append(self.Block(channels[i + 1] + channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    @staticmethod
    def _pool_if_possible(x):
        if x.shape[-2] >= 4 and x.shape[-1] >= 4:
            return F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x):
        valid = ~torch.isnan(x)
        valid_out = valid if self.out_channels == self.in_channels else valid[:, :1]
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for block in self.encoder:
            h = self._pool_if_possible(h)
            h = block(h)
            skips.append(h)

        h = self._pool_if_possible(h)
        h = self.bottleneck(h)

        for gate, block, skip in zip(self.detail_gates, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = gate(h, skip)
            h = block(torch.cat([h, skip], dim=1))

        out = self.head(h)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


