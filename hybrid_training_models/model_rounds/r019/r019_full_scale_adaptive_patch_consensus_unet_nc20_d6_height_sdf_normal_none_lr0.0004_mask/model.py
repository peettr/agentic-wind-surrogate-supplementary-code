import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    groups = min(8, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class scale_adaptive_patch_consensus_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class PatchConsensus(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.fuse = nn.Sequential(
                nn.Conv2d(channels * 4, channels, kernel_size=1, padding=0, bias=False),
                _gn(channels),
                nn.SiLU(inplace=True),
            )

        def _pooled(self, x, k):
            h, w = x.shape[-2:]
            kk = max(1, min(k, h, w))
            if kk <= 1:
                return x
            y = F.avg_pool2d(x, kernel_size=kk, stride=kk)
            return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

        def forward(self, x):
            p1 = x
            p2 = self._pooled(x, 2)
            p4 = self._pooled(x, 4)
            p8 = self._pooled(x, 8)
            return self.fuse(torch.cat([p1, p2, p4, p8], dim=1))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.PatchConsensus(channels[-1]),
        )

        self.decoders = nn.ModuleList()
        self.up_fuse = nn.ModuleList()
        dec_in = channels[-1]
        for skip_ch in reversed(channels):
            self.up_fuse.append(
                nn.Sequential(
                    nn.Conv2d(dec_in + skip_ch, skip_ch, kernel_size=1, padding=0, bias=False),
                    _gn(skip_ch),
                    nn.SiLU(inplace=True),
                )
            )
            self.decoders.append(self.ConvBlock(skip_ch, skip_ch))
            dec_in = skip_ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i < self.depth - 1:
                if y.shape[-1] >= 2 and y.shape[-2] >= 2:
                    y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for fuse, decoder, skip in zip(self.up_fuse, self.decoders, reversed(skips)):
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = fuse(y)
            y = decoder(y)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid[:, :1].expand(-1, y.shape[1], -1, -1)
        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y