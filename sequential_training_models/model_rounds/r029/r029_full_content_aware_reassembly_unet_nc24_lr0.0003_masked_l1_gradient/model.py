import torch
import torch.nn as nn
import torch.nn.functional as F

class content_aware_reassembly_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, k=3):
            super().__init__()
            self.pad = nn.ReflectionPad2d(k // 2)
            self.conv = nn.Conv2d(in_ch, out_ch, k, padding=0)

        def forward(self, x):
            return self.conv(self.pad(x))

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                content_aware_reassembly_unet.RefConv(in_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                content_aware_reassembly_unet.RefConv(out_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
            self.skip = nn.Identity() if in_ch == out_ch else content_aware_reassembly_unet.RefConv(in_ch, out_ch, 1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class CARAFEUp(nn.Module):
        def __init__(self, ch, scale=2, kernel_size=3):
            super().__init__()
            self.scale = scale
            self.kernel_size = kernel_size
            hidden = max(ch // 4, 16)
            self.encoder = nn.Sequential(
                content_aware_reassembly_unet.RefConv(ch, hidden, 1),
                nn.SiLU(inplace=True),
                content_aware_reassembly_unet.RefConv(hidden, (scale * scale) * (kernel_size * kernel_size), 3),
            )
            self.pad = nn.ReflectionPad2d(kernel_size // 2)

        def forward(self, x):
            b, c, h, w = x.shape
            s = self.scale
            k = self.kernel_size

            weights = self.encoder(x)
            weights = F.pixel_shuffle(weights, s)
            weights = weights.view(b, k * k, h * s, w * s)
            weights = F.softmax(weights, dim=1)

            patches = F.unfold(self.pad(x), kernel_size=k, padding=0)
            patches = patches.view(b, c, k * k, h, w)
            patches = patches.repeat_interleave(s, dim=3).repeat_interleave(s, dim=4)

            return (patches * weights.unsqueeze(1)).sum(dim=2)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self.Block(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(2)
        self.bottleneck = self.Block(channels[-1], channels[-1])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.ups.append(self.CARAFEUp(channels[i + 1], scale=2, kernel_size=3))
            self.decoders.append(self.Block(channels[i + 1] + channels[i], channels[i]))

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.out_conv(self.out_pad(h))
        out = torch.where(valid[:, :1], out, torch.full_like(out, float("nan")))
        return out