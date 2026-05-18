import torch
import torch.nn as nn
import torch.nn.functional as F

class coarse_conditioned_detail_decoder_unet(nn.Module):
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

    class Downsample(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.op = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0, bias=False),
            )

        def forward(self, x):
            return self.op(x)

    class Upsample(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.proj = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x, size):
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            return self.proj(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_proj = self.ConvBlock(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(self.Downsample(channels[i - 1]))
            self.encoders.append(self.ConvBlock(channels[i - 1], channels[i]))

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.ups.append(self.Upsample(channels[i + 1], channels[i]))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.coarse_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(min(8, channels[-1]), channels[-1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[-1], out_channels, kernel_size=1, padding=0),
        )

        self.detail_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.input_proj(x_masked)
        skips.append(h)

        for down, encoder in zip(self.downs, self.encoders):
            h = down(h)
            h = encoder(h)
            skips.append(h)

        h = self.bottleneck(h)

        coarse = self.coarse_head(h)
        coarse = F.interpolate(coarse, size=x.shape[-2:], mode="bilinear", align_corners=False)

        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            h = up(h, skip.shape[-2:])
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        detail = self.detail_head(h)
        output = coarse + detail
        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output