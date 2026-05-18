import torch
import torch.nn as nn
import torch.nn.functional as F

class beno_multitoken_film_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = self._ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        self.projections = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            up_ch = channels[i + 1]
            skip_ch = channels[i]
            self.projections.append(nn.Conv2d(up_ch, skip_ch, 1))
            self.decoders.append(self._ConvBlock(skip_ch * 2, skip_ch))

        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for decoder, projection, skip in zip(self.decoders, self.projections, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = projection(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.final(h)

        if valid.shape[1] != output.shape[1]:
            valid = valid[:, :1].expand(-1, output.shape[1], -1, -1)

        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output