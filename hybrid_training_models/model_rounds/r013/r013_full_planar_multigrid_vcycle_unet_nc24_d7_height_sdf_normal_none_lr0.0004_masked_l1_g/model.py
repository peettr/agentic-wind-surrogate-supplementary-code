import torch
import torch.nn as nn
import torch.nn.functional as F

class planar_multigrid_vcycle_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                planar_multigrid_vcycle_unet.ReflectConv(in_ch, out_ch),
                planar_multigrid_vcycle_unet.ReflectConv(out_ch, out_ch),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=7):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self.Block(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(self.Block(channels[i - 1], channels[i]))

        self.bottleneck = self.Block(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self.Block(channels[i + 1] + channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for block in self.encoder:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            h = block(h)
            skips.append(h)

        h = self.bottleneck(h)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(out)

        out = out.clone()
        out[~valid] = float("nan")
        return out


