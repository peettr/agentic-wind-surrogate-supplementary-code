import torch
import torch.nn as nn
import torch.nn.functional as F

class hrnet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        self.up_reduce = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_reduce.append(
                nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(channels[i + 1], channels[i], kernel_size=3, padding=0, bias=False),
                    nn.BatchNorm2d(channels[i]),
                    nn.GELU(),
                )
            )
            self.decoder.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        valid_out = valid.all(dim=1, keepdim=True)
        x = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x

        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i != self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for reduce_block, decode_block, skip in zip(self.up_reduce, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = reduce_block(h)
            h = torch.cat([h, skip], dim=1)
            h = decode_block(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out