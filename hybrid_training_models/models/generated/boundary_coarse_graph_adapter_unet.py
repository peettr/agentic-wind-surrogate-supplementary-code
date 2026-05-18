import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class boundary_coarse_graph_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(cin, cout, kernel_size=3, padding=0, bias=False),
                _gn(cout),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(cout, cout, kernel_size=3, padding=0, bias=False),
                _gn(cout),
                nn.GELU(),
            )

        self.encoder = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoder.append(conv_block(prev, ch))
            prev = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=0, bias=False),
            _gn(bottleneck_ch),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=0, bias=False),
            _gn(bottleneck_ch),
            nn.GELU(),
        )

        self.up_reduce = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            self.up_reduce.append(
                nn.Sequential(
                    nn.ReflectionPad2d(0),
                    nn.Conv2d(channels[i + 1], channels[i], kernel_size=1, padding=0, bias=False),
                    _gn(channels[i]),
                    nn.GELU(),
                )
            )
            self.decoder.append(conv_block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.GELU(),
            nn.ReflectionPad2d(0),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i != self.depth - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_reduce, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        output = self.head(h)

        if valid.shape != output.shape:
            valid = valid.all(dim=1, keepdim=True).expand_as(output)

        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output


