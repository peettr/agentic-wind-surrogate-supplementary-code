import torch
import torch.nn as nn
import torch.nn.functional as F

class height_geomlift_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                height_geomlift_unet.RefConv(in_ch, out_ch),
                height_geomlift_unet.RefConv(out_ch, out_ch),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_ch = in_channels + 2
        for ch in channels:
            self.encoder.append(self.Block(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = self.Block(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        self.up_proj = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_proj.append(nn.Conv2d(prev_ch, channels[i], kernel_size=1, bias=False))
            self.decoder.append(self.Block(channels[i] * 2, channels[i]))
            prev_ch = channels[i]

        self.head = nn.Sequential(
            self.RefConv(prev_ch, n_c),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_c, out_channels, kernel_size=3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        b, _, h, w = x_masked.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        y = torch.cat([x_masked, xx, yy], dim=1)

        skips = []
        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i != len(self.encoder) - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for proj, block, skip in zip(self.up_proj, self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = proj(y)
            y = block(torch.cat([y, skip], dim=1))

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != y.shape[1]:
            out_valid = out_valid.expand(-1, y.shape[1], -1, -1)
        y = torch.where(out_valid, y, torch.full_like(y, float("nan")))
        return y