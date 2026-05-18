import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class CenterRouter(nn.Module):
    def __init__(self, channels, experts=4):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, 3, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.GroupNorm(min(8, channels), channels),
                nn.SiLU(inplace=True),
            )
            for _ in range(experts)
        ])
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 4, 1), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // 4, 1), experts, 1),
        )
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        weights = torch.softmax(self.router(x), dim=1)
        y = 0
        for i, expert in enumerate(self.experts):
            y = y + expert(x) * weights[:, i:i + 1]
        return self.fuse(x + y)

class shared_expert_center_router_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], channels[-1]),
            CenterRouter(channels[-1]),
            ConvBlock(channels[-1], channels[-1]),
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            ch = channels[i]
            self.upconvs.append(nn.ConvTranspose2d(prev, ch, 2, stride=2))
            self.decoders.append(ConvBlock(ch + ch, ch))
            prev = ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = self.down(h)

        h = self.bottleneck(h)

        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = decoder(torch.cat([h, skip], dim=1))

        output = self.head(h)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return output.masked_fill(~valid.expand(-1, output.shape[1], -1, -1), float("nan"))


