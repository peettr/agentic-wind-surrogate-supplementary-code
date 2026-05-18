import torch
import torch.nn as nn
import torch.nn.functional as F

class boundary_film_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class DownBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            self.conv = boundary_film_unet.ConvBlock(in_ch, out_ch)

        def forward(self, x):
            return self.conv(self.pool(x))

    class UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
            self.conv = boundary_film_unet.ConvBlock(out_ch + skip_ch, out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=32, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.ConvBlock(in_channels + 1, channels[0])
        self.encoder = nn.ModuleList(
            self.DownBlock(channels[i - 1], channels[i]) for i in range(1, depth)
        )

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList(
            self.UpBlock(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        )

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        mask = valid.to(dtype=x.dtype)
        h = torch.cat([x_masked, mask], dim=1)

        skips = []
        h = self.stem(h)
        skips.append(h)

        for down in self.encoder:
            h = down(h)
            skips.append(h)

        h = self.bottleneck(h)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            h = up(h, skip)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid = valid[:, :1].expand(-1, out.shape[1], -1, -1)

        out = torch.where(valid, out, torch.full_like(out, float("nan")))
        return out