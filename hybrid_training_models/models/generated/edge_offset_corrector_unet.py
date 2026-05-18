import torch
import torch.nn as nn
import torch.nn.functional as F

class edge_offset_corrector_unet(nn.Module):
    class _ConvBlock(nn.Module):
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

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        self.reduce = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.reduce.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1, padding=0, bias=False))
            self.decoders.append(self._ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i != self.depth - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for reduce, decoder, skip in zip(self.reduce, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = reduce(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid.all(dim=1, keepdim=True)
        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out