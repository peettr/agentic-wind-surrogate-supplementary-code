import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(channels):
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False),
            nn.GroupNorm(_gn_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.skip(x)


class _Adapter(nn.Module):
    def __init__(self, channels, bottleneck):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.up = nn.Conv2d(bottleneck, channels, 1)

    def forward(self, x):
        return x + self.up(F.silu(self.down(x), inplace=True))


class sparse_shared_projection_expert_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(_Block(prev, ch))
            prev = ch

        bottleneck_channels = channels[-1]
        self.bottleneck = _Block(bottleneck_channels, bottleneck_channels)

        adapter_channels = max(1, bottleneck_channels // 4)
        self.shared_adapter = _Adapter(bottleneck_channels, adapter_channels)

        self.up_projections = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for skip_ch in reversed(channels[:-1]):
            self.up_projections.append(nn.Conv2d(prev, skip_ch, 1))
            self.decoders.append(_Block(skip_ch * 2, skip_ch))
            prev = skip_ch

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i < len(self.encoders) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)
        h = self.shared_adapter(h)

        for up, decoder, skip in zip(self.up_projections, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.head(h)
        output = torch.where(valid[:, :1], output, torch.full_like(output, float("nan")))
        return output