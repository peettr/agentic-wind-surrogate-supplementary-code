import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels, max_groups=8):
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectionConv2d(in_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            _ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _CenterAdapter(nn.Module):
    def __init__(self, channels, num_experts=4):
        super().__init__()
        hidden = max(channels // 4, 1)
        self.experts = nn.ModuleList([
            nn.Sequential(
                _ReflectionConv2d(channels, channels, 3, bias=False),
                _gn(channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, num_experts, 1),
        )
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=1)
        y = 0
        for i, expert in enumerate(self.experts):
            y = y + expert(x) * weights[:, i:i + 1]
        return x + self.mix(y)


class shared_expert_center_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(_ConvBlock(prev, ch))
            prev = ch

        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            _ConvBlock(channels[-1], channels[-1]),
            _CenterAdapter(channels[-1]),
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(channels[i + 1], channels[i], 1))
            self.decoders.append(_ConvBlock(channels[i] * 2, channels[i]))

        self.out_head = nn.Sequential(
            _ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i < len(self.encoders) - 1:
                h = self.downsample(h)

        h = self.bottleneck(h)

        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_conv(h)
            h = decoder(torch.cat([h, skip], dim=1))

        output = self.out_head(h)
        output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if output.shape[1] == valid.shape[1]:
            output = output.masked_fill(~valid, float("nan"))
        else:
            output = output.masked_fill(~valid.any(dim=1, keepdim=True), float("nan"))

        return output


