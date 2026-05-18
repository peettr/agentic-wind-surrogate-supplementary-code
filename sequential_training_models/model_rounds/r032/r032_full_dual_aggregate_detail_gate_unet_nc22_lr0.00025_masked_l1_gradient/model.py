import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


class dual_aggregate_detail_gate_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0, bias=bias),
            )

        def forward(self, x):
            return self.net(x)

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = dual_aggregate_detail_gate_unet.ReflectConv(in_ch, out_ch)
            self.norm1 = nn.GroupNorm(_gn(out_ch), out_ch)
            self.conv2 = dual_aggregate_detail_gate_unet.ReflectConv(out_ch, out_ch)
            self.norm2 = nn.GroupNorm(_gn(out_ch), out_ch)
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(
                in_ch, out_ch, kernel_size=1, padding=0, bias=False
            )

        def forward(self, x):
            residual = self.skip(x)
            x = F.silu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return F.silu(x + residual)

    class DetailGate(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.local = nn.Sequential(
                dual_aggregate_detail_gate_unet.ReflectConv(ch, ch),
                nn.GroupNorm(_gn(ch), ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, ch, kernel_size=1, padding=0),
            )
            mid_ch = max(ch // 4, 1)
            self.context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, mid_ch, kernel_size=1, padding=0),
                nn.SiLU(inplace=True),
                nn.Conv2d(mid_ch, ch, kernel_size=1, padding=0),
            )

        def forward(self, x):
            return x * torch.sigmoid(self.local(x) + self.context(x))

    class UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
            self.skip_gate = dual_aggregate_detail_gate_unet.DetailGate(skip_ch)
            self.aggregate_gate = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch + skip_ch, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
            self.block = dual_aggregate_detail_gate_unet.ConvBlock(out_ch + skip_ch, out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.proj(x)
            skip = self.skip_gate(skip)
            merged = torch.cat([x, skip], dim=1)
            merged = merged * self.aggregate_gate(merged)
            return self.block(merged)

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

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.DetailGate(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self.UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            self.DetailGate(channels[0]),
            self.ReflectConv(channels[0], channels[0]),
            nn.GroupNorm(_gn(channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            y = decoder(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid.any(dim=1, keepdim=True)
        else:
            valid_out = valid

        return torch.where(valid_out, y, torch.full_like(y, float("nan")))