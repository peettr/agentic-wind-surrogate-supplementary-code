import torch
import torch.nn as nn
import torch.nn.functional as F

class mean_residual_decoder_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
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
            return self.block(x)

    class Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.pad = nn.ReflectionPad2d(1)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=0, bias=False)
            self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
            self.act = nn.SiLU(inplace=True)
            self.block = mean_residual_decoder_unet.ConvBlock(out_ch, out_ch)

        def forward(self, x):
            x = self.pad(x)
            x = self.conv(x)
            x = self.norm(x)
            x = self.act(x)
            return self.block(x)

    class Up(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
            self.block = mean_residual_decoder_unet.ConvBlock(out_ch + skip_ch, out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            return self.block(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        max_ch = n_c * 8
        channels = [min(n_c * (2 ** i), max_ch) for i in range(depth)]

        self.stem = self.ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList([
            self.Down(channels[i - 1], channels[i]) for i in range(1, depth)
        ])

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList([
            self.Up(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        ])

        self.residual_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

        self.mean_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[-1], channels[-1], kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[-1], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for down in self.encoder:
            h = down(h)
            skips.append(h)

        h = self.bottleneck(h)
        mean = self.mean_head(h)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            h = up(h, skip)

        residual = self.residual_head(h)
        output = residual + mean
        output = torch.where(valid[:, :output.shape[1]], output, torch.full_like(output, float("nan")))
        return output