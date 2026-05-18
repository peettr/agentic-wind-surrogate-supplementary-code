import torch
import torch.nn as nn
import torch.nn.functional as F

class shared_norm_cross_shape_adapter_unet(nn.Module):
    class ReflectConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            groups = min(8, out_channels)
            while out_channels % groups != 0:
                groups -= 1
            self.net = nn.Sequential(
                shared_norm_cross_shape_adapter_unet.ReflectConv2d(in_channels, out_channels, 3, bias=False),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
                shared_norm_cross_shape_adapter_unet.ReflectConv2d(out_channels, out_channels, 3, bias=False),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
            )
            self.skip = nn.Identity() if in_channels == out_channels else nn.Sequential(
                shared_norm_cross_shape_adapter_unet.ReflectConv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(groups, out_channels),
            )

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class ShapeAdapter(nn.Module):
        def __init__(self, channels):
            super().__init__()
            hidden = max(8, channels // 4)
            self.local = nn.Sequential(
                shared_norm_cross_shape_adapter_unet.ReflectConv2d(channels, hidden, 1, bias=True),
                nn.SiLU(inplace=True),
                shared_norm_cross_shape_adapter_unet.ReflectConv2d(hidden, channels, 1, bias=True),
                nn.Sigmoid(),
            )
            self.context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return x * (1.0 + self.local(x) * self.context(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        self.adapters = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev, ch))
            self.adapters.append(self.ShapeAdapter(ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.ShapeAdapter(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            in_ch = channels[i + 1]
            skip_ch = channels[i]
            self.up_projs.append(self.ReflectConv2d(in_ch, skip_ch, 1, bias=False))
            self.decoders.append(self.ConvBlock(skip_ch * 2, skip_ch))

        self.head = nn.Sequential(
            self.ReflectConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            self.ReflectConv2d(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, encoder in enumerate(self.encoders):
            h = self.adapters[i](encoder(h))
            skips.append(h)
            if i != self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = decoder(torch.cat([h, skip], dim=1))

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] == out.shape[1]:
            out_valid = valid
        else:
            out_valid = valid.all(dim=1, keepdim=True).expand(-1, out.shape[1], -1, -1)
        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out


