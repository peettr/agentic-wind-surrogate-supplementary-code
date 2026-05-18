import torch
import torch.nn as nn
import torch.nn.functional as F

class unet_v2_baseline(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=48, depth=6):
        super().__init__()

        self.depth = depth
        self.out_channels = out_channels
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        class ConvBlock(nn.Module):
            def __init__(self, c_in, c_out):
                super().__init__()
                self.net = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=0, bias=False),
                    nn.GroupNorm(min(8, c_out), c_out),
                    nn.SiLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(c_out, c_out, kernel_size=3, padding=0, bias=False),
                    nn.GroupNorm(min(8, c_out), c_out),
                    nn.SiLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch

        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = channels[i] if i == depth - 1 else channels[i + 1]
            skip_ch = channels[i]
            out_ch = channels[i]
            self.up_projs.append(nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            ))
            self.decoders.append(ConvBlock(out_ch + skip_ch, out_ch))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for i, (up_proj, dec) in enumerate(zip(self.up_projs, self.decoders)):
            skip = skips[-(i + 1)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if self.out_channels != x.shape[1]:
            out_valid = valid[:, :1].expand(-1, self.out_channels, -1, -1)

        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out