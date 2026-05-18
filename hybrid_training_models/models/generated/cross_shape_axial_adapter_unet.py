import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels, max_groups=8):
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class cross_shape_axial_adapter_unet(nn.Module):
    class ReflectConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.proj = (
                cross_shape_axial_adapter_unet.ReflectConv2d(in_channels, out_channels, 1, bias=False)
                if in_channels != out_channels else nn.Identity()
            )
            self.net = nn.Sequential(
                cross_shape_axial_adapter_unet.ReflectConv2d(in_channels, out_channels, 3, bias=False),
                _gn(out_channels),
                nn.SiLU(inplace=True),
                cross_shape_axial_adapter_unet.ReflectConv2d(out_channels, out_channels, 3, bias=False),
                _gn(out_channels),
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.proj(x))

    class CrossShapeAxialAdapter(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.h_pad = nn.ReflectionPad2d((3, 3, 0, 0))
            self.v_pad = nn.ReflectionPad2d((0, 0, 3, 3))
            self.h_conv = nn.Conv2d(channels, channels, (1, 7), padding=0, groups=channels, bias=False)
            self.v_conv = nn.Conv2d(channels, channels, (7, 1), padding=0, groups=channels, bias=False)
            self.mix = cross_shape_axial_adapter_unet.ReflectConv2d(channels, channels, 1, bias=False)
            self.norm = _gn(channels)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(1, channels // 4), 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(1, channels // 4), channels, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            y = self.h_conv(self.h_pad(x)) + self.v_conv(self.v_pad(x))
            y = self.mix(y)
            y = self.norm(y)
            return x + y * self.gate(y)

    class UpBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.reduce = cross_shape_axial_adapter_unet.ReflectConv2d(
                in_channels + skip_channels, out_channels, 1, bias=False
            )
            self.block = cross_shape_axial_adapter_unet.ConvBlock(out_channels, out_channels)
            self.adapter = cross_shape_axial_adapter_unet.CrossShapeAxialAdapter(out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.reduce(x)
            x = self.block(x)
            return self.adapter(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self.ConvBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(depth):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            self.encoder.append(nn.Sequential(
                self.ConvBlock(in_ch, out_ch),
                self.CrossShapeAxialAdapter(out_ch),
            ))

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.CrossShapeAxialAdapter(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self.UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ConvBlock(channels[0], channels[0]),
            self.ReflectConv2d(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        orig_size = x_masked.shape[-2:]
        h = self.stem(x_masked)

        skips = []
        for i, enc in enumerate(self.encoder):
            h = enc(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoder, reversed(skips[:-1])):
            h = dec(h, skip)

        if h.shape[-2:] != orig_size:
            h = F.interpolate(h, size=orig_size, mode="bilinear", align_corners=False)

        out = self.head(h)
        valid_out = valid
        if valid_out.shape[1] != out.shape[1]:
            valid_out = valid_out[:, :1].expand_as(out)
        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out