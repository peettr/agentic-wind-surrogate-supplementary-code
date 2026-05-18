import torch
import torch.nn as nn
import torch.nn.functional as F

class high_preservation_dual_aggregation_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self._conv_block(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(nn.ModuleDict({
                "down": nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=3, stride=2, padding=0, bias=False),
                    nn.GroupNorm(self._groups(channels[i]), channels[i]),
                    nn.SiLU(inplace=True),
                ),
                "block": self._conv_block(channels[i], channels[i]),
            }))

        self.bottleneck = nn.Sequential(
            self._conv_block(channels[-1], channels[-1]),
            self._conv_block(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            in_ch = channels[i + 1]
            skip_ch = channels[i]
            out_ch = channels[i]
            self.decoder.append(nn.ModuleDict({
                "skip_proj": nn.Sequential(
                    nn.ReflectionPad2d(0),
                    nn.Conv2d(skip_ch, out_ch, kernel_size=1, padding=0, bias=False),
                    nn.GroupNorm(self._groups(out_ch), out_ch),
                    nn.SiLU(inplace=True),
                ),
                "up_proj": nn.Sequential(
                    nn.ReflectionPad2d(0),
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False),
                    nn.GroupNorm(self._groups(out_ch), out_ch),
                    nn.SiLU(inplace=True),
                ),
                "fuse": self._conv_block(out_ch * 2, out_ch),
                "refine": self._conv_block(out_ch, out_ch),
            }))

        self.high_preserve = nn.Sequential(
            self._conv_block(in_channels, channels[0]),
            self._conv_block(channels[0], channels[0]),
        )

        self.head = nn.Sequential(
            self._conv_block(channels[0] * 2, channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    @staticmethod
    def _groups(channels):
        for g in (8, 4, 2):
            if channels % g == 0:
                return g
        return 1

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(self._groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(self._groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for level in self.encoder:
            h = level["down"](h)
            h = level["block"](h)
            skips.append(h)

        h = self.bottleneck(h)

        for idx, level in enumerate(self.decoder):
            skip = skips[-(idx + 2)]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            h_up = level["up_proj"](h)
            h_skip = level["skip_proj"](skip)

            avg = 0.5 * (h_up + h_skip)
            cat = torch.cat([avg, h_skip], dim=1)

            h = level["fuse"](cat)
            h = level["refine"](h + avg)

        high = self.high_preserve(x_masked)
        if h.shape[-2:] != high.shape[-2:]:
            h = F.interpolate(h, size=high.shape[-2:], mode="bilinear", align_corners=False)

        out = self.head(torch.cat([h, high], dim=1))

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            out_valid = out_valid.all(dim=1, keepdim=True).expand_as(out)

        out = torch.where(out_valid, out, torch.full_like(out, torch.nan))
        return out


