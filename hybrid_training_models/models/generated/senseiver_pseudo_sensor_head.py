import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for num_groups in range(min(8, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class senseiver_pseudo_sensor_head(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        def conv3x3(cin, cout):
            return nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(cin, cout, kernel_size=3, bias=False),
                _gn(cout),
                nn.SiLU(inplace=True),
            )

        class Block(nn.Module):
            def __init__(self, cin, cout):
                super().__init__()
                self.net = nn.Sequential(
                    conv3x3(cin, cout),
                    conv3x3(cout, cout),
                )
                self.skip = nn.Conv2d(cin, cout, kernel_size=1, bias=False) if cin != cout else nn.Identity()

            def forward(self, x):
                return self.net(x) + self.skip(x)

        self.encoder = nn.ModuleList()
        prev = in_channels
        for c in channels:
            self.encoder.append(Block(prev, c))
            prev = c

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = Block(channels[-1], channels[-1])

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up.append(nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.decoder.append(Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i != self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for up, block, skip in zip(self.up, self.decoder, reversed(skips[:-1])):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = torch.where(valid.expand_as(out), out, torch.full_like(out, float("nan")))
        return out