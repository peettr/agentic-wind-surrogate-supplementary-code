import torch
import torch.nn as nn
import torch.nn.functional as F

class boundary_ring_mixer_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                boundary_ring_mixer_unet.RefConv(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                boundary_ring_mixer_unet.RefConv(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class BoundaryRingMixer(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.local = nn.Sequential(
                boundary_ring_mixer_unet.RefConv(channels, channels, 3, bias=False),
                nn.GroupNorm(min(8, channels), channels),
                nn.SiLU(inplace=True),
            )
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels, 1, padding=0),
                nn.Sigmoid(),
            )

        def forward(self, x):
            h, w = x.shape[-2:]
            yy = torch.arange(h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
            xx = torch.arange(w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
            dist = torch.minimum(torch.minimum(yy, h - 1 - yy), torch.minimum(xx, w - 1 - xx))
            ring = torch.exp(-dist / max(1.0, min(h, w) * 0.035))
            mixed = self.local(x * ring)
            return x + mixed * self.gate(x)

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
        self.bottleneck = nn.Sequential(
            self.Block(channels[-1], channels[-1]),
            self.BoundaryRingMixer(channels[-1]),
            self.Block(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0, bias=False))
            self.decoders.append(self.Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.RefConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid = valid[:, :1].expand(-1, out.shape[1], -1, -1)
        out = out.masked_fill(~valid, float("nan"))
        return out