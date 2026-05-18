import torch
import torch.nn as nn
import torch.nn.functional as F

class dual_spatial_channel_agg_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class DSCA(nn.Module):
        def __init__(self, ch):
            super().__init__()
            hidden = max(ch // 8, 4)
            self.channel = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, hidden, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, ch, 1),
                nn.Sigmoid(),
            )
            self.spatial = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(2, 1, 7, padding=0),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = x * self.channel(x)
            avg = torch.mean(x, dim=1, keepdim=True)
            mx = torch.amax(x, dim=1, keepdim=True)
            x = x * self.spatial(torch.cat([avg, mx], dim=1))
            return x

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = dual_spatial_channel_agg_unet.RefConv(in_ch, out_ch)
            self.conv2 = dual_spatial_channel_agg_unet.RefConv(out_ch, out_ch)
            self.attn = dual_spatial_channel_agg_unet.DSCA(out_ch)
            self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

        def forward(self, x):
            y = self.conv1(x)
            y = self.conv2(y)
            y = self.attn(y)
            return y + self.skip(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = []
        for i in range(depth):
            channels.append(min(n_c * (2 ** i), n_c * 8))
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self.Block(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(2)
        self.bottleneck = self.Block(channels[-1], channels[-1])

        self.up_reduce = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_reduce.append(nn.Conv2d(channels[i + 1], channels[i], 1, bias=False))
            self.decoders.append(self.Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.RefConv(channels[0], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked
        for i, enc in enumerate(self.encoders):
            y = enc(y)
            skips.append(y)
            if i != self.depth - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for reduce, dec, skip in zip(self.up_reduce, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = reduce(y)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)
        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y