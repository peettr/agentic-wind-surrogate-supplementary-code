import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = min(8, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class multiwavelet_vcycle_detail_adapter_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                multiwavelet_vcycle_detail_adapter_unet.ReflectConv(in_ch, out_ch, 3),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                multiwavelet_vcycle_detail_adapter_unet.ReflectConv(out_ch, out_ch, 3),
                _gn(out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class DetailAdapter(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.low = nn.Sequential(
                nn.AvgPool2d(2, 2),
                multiwavelet_vcycle_detail_adapter_unet.ReflectConv(ch, ch, 3),
                _gn(ch),
                nn.SiLU(inplace=True),
            )
            self.detail = nn.Sequential(
                multiwavelet_vcycle_detail_adapter_unet.ReflectConv(ch, ch, 3),
                _gn(ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, ch, 1),
            )

        def forward(self, x):
            low = self.low(x)
            up = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return x + self.detail(x - up)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self.ConvBlock(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        self.adapters = nn.ModuleList()
        for i in range(1, depth):
            self.encoders.append(self.ConvBlock(channels[i - 1], channels[i]))
            self.adapters.append(self.DetailAdapter(channels[i]))

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.DetailAdapter(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(channels[i + 1], channels[i], 1))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for encoder, adapter in zip(self.encoders, self.adapters):
            h = F.avg_pool2d(h, 2, 2)
            h = encoder(h)
            h = adapter(h)
            skips.append(h)

        h = self.bottleneck(h)

        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_conv(h)
            h = decoder(torch.cat([h, skip], dim=1))

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != self.out_channels:
            valid = valid.all(dim=1, keepdim=True).expand(-1, self.out_channels, -1, -1)

        return torch.where(valid, out, torch.full_like(out, float("nan")))