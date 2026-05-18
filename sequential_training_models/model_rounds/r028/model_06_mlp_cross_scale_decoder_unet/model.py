import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp_cross_scale_decoder_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                mlp_cross_scale_decoder_unet.ReflectConv(in_channels, out_channels, 3),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                mlp_cross_scale_decoder_unet.ReflectConv(out_channels, out_channels, 3),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class CrossScaleMLP(nn.Module):
        def __init__(self, channels):
            super().__init__()
            hidden = min(channels * 2, channels + 256)
            self.local = nn.Sequential(
                mlp_cross_scale_decoder_unet.ReflectConv(channels, channels, 3),
                nn.GroupNorm(min(8, channels), channels),
                nn.SiLU(inplace=True),
            )
            self.mlp = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
            )
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(channels // 4, 1), kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(channels // 4, 1), channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            y = self.local(x)
            y = self.mlp(y) * self.gate(y)
            return x + y

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
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

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.CrossScaleMLP(channels[-1]),
            self.CrossScaleMLP(channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoder.append(
                nn.Sequential(
                    self.ConvBlock(channels[i] * 2, channels[i]),
                    self.CrossScaleMLP(channels[i]),
                )
            )

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3),
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
            if i != self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up_proj, dec, skip in zip(self.up_projs, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != out.shape[1]:
            valid_out = valid_out.expand(-1, out.shape[1], -1, -1)

        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


