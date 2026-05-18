import torch
import torch.nn as nn
import torch.nn.functional as F

class latent_graph_processor_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class Bottleneck(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.local = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            y = self.local(x)
            y = y * self.context(y)
            return self.act(x + y)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.Bottleneck(channels[-1])

        self.up_reduce = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_reduce.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoder.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3),
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
            if i < self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for reduce, block, skip in zip(self.up_reduce, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = reduce(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid_out = valid[:, :1].expand(-1, out.shape[1], -1, -1)
        else:
            valid_out = valid

        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


