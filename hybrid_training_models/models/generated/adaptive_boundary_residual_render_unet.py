import torch
import torch.nn as nn
import torch.nn.functional as F

class adaptive_boundary_residual_render_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.block = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class ResidualBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = adaptive_boundary_residual_render_unet.RefConv(in_ch, out_ch)
            self.pad2 = nn.ReflectionPad2d(1)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=0, bias=False)
            self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.act = nn.SiLU(inplace=True)

            if in_ch != out_ch:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                )
            else:
                self.skip = nn.Identity()

        def forward(self, x):
            y = self.conv1(x)
            y = self.pad2(y)
            y = self.conv2(y)
            y = self.norm2(y)
            return self.act(y + self.skip(x))

    class BoundaryGate(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, 3, padding=0, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, padding=0),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * (1.0 + self.net(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.channels = channels

        self.stem = self.ResidualBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()

        for i in range(depth):
            in_ch = channels[0] if i == 0 else channels[i - 1]
            self.encoder.append(self.ResidualBlock(in_ch, channels[i]))
            if i < depth - 1:
                self.down.append(nn.AvgPool2d(kernel_size=2, stride=2))

        self.bottleneck = nn.Sequential(
            self.ResidualBlock(channels[-1], channels[-1]),
            self.BoundaryGate(channels[-1]),
            self.ResidualBlock(channels[-1], channels[-1]),
        )

        self.up_reduce = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            self.up_reduce.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0, bias=False))
            self.decoder.append(self.ResidualBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ResidualBlock(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)

        for i, enc in enumerate(self.encoder):
            y = enc(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.down[i](y)

        y = self.bottleneck(y)

        for reduce, dec, skip in zip(self.up_reduce, self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = reduce(y)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels == self.in_channels:
            y = y + x_masked

        out_valid = valid
        if out_valid.shape[1] != y.shape[1]:
            out_valid = out_valid.any(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)

        y = torch.where(out_valid, y, torch.full_like(y, float("nan")))
        return y