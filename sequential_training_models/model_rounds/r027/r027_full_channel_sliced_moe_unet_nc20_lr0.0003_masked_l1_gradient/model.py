import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class channel_sliced_moe_unet(nn.Module):
    class RefConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                channel_sliced_moe_unet.RefConv2d(in_ch, out_ch, 3, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                channel_sliced_moe_unet.RefConv2d(out_ch, out_ch, 3, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class ChannelSlicedMoE(nn.Module):
        def __init__(self, channels, experts=4):
            super().__init__()
            experts = max(1, min(experts, channels))
            self.experts = experts
            self.convs = nn.ModuleList()
            base = channels // experts
            rem = channels % experts
            self.sizes = []
            for i in range(experts):
                c = base + (1 if i < rem else 0)
                self.sizes.append(c)
                self.convs.append(nn.Sequential(
                    channel_sliced_moe_unet.RefConv2d(c, c, 3, bias=False),
                    nn.GroupNorm(1, c),
                    nn.SiLU(inplace=True),
                    channel_sliced_moe_unet.RefConv2d(c, c, 3, bias=False),
                ))
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, experts, 1),
                nn.Softmax(dim=1),
            )
            self.proj = nn.Sequential(
                channel_sliced_moe_unet.RefConv2d(channels, channels, 3, bias=False),
                _gn(channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            weights = self.gate(x)
            chunks = torch.split(x, self.sizes, dim=1)
            outs = []
            for i, chunk in enumerate(chunks):
                outs.append(self.convs[i](chunk) * weights[:, i:i + 1])
            return self.proj(torch.cat(outs, dim=1) + x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoder = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoder.append(self.ConvBlock(prev, ch))
            prev = ch

        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.ChannelSlicedMoE(channels[-1], experts=4),
        )

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for ch in reversed(channels[:-1]):
            self.up.append(nn.ConvTranspose2d(prev, ch, kernel_size=2, stride=2))
            self.decoder.append(self.ConvBlock(ch + ch, ch))
            prev = ch

        self.head = nn.Sequential(
            self.RefConv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i != len(self.encoder) - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        skips = skips[:-1][::-1]
        for up, block, skip in zip(self.up, self.decoder, skips):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = block(torch.cat([h, skip], dim=1))

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] == out.shape[1]:
            out = out.masked_fill(~valid, float("nan"))
        else:
            out_valid = valid.all(dim=1, keepdim=True).expand(-1, out.shape[1], -1, -1)
            out = out.masked_fill(~out_valid, float("nan"))

        return out