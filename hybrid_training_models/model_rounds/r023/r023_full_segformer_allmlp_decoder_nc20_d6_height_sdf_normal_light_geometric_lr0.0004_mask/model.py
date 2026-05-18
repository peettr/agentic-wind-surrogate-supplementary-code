import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class segformer_allmlp_decoder(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                _gn(out_ch),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                _gn(out_ch),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class DownBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.down = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=0, bias=False),
                _gn(out_ch),
                nn.GELU(),
            )
            self.block = segformer_allmlp_decoder.ConvBlock(out_ch, out_ch)

        def forward(self, x):
            return self.block(self.down(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [n_c * min(2 ** i, 8) for i in range(depth)]

        self.out_channels = out_channels
        self.stem = self.ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            self.DownBlock(channels[i - 1], channels[i]) for i in range(1, depth)
        )

        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=0, bias=False),
            _gn(bottleneck_ch),
            nn.GELU(),
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=1, padding=0, bias=False),
            _gn(bottleneck_ch),
            nn.GELU(),
        )

        self.proj = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(ch, channels[0], kernel_size=1, padding=0, bias=False),
                _gn(channels[0]),
                nn.GELU(),
            )
            for ch in channels
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels[0] * depth, channels[0] * 4, kernel_size=1, padding=0, bias=False),
            _gn(channels[0] * 4),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0] * 4, channels[0] * 2, kernel_size=3, padding=0, bias=False),
            _gn(channels[0] * 2),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0] * 2, channels[0], kernel_size=3, padding=0, bias=False),
            _gn(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        feats = []
        y = self.stem(x_masked)
        feats.append(y)

        for block in self.encoder:
            y = block(y)
            feats.append(y)

        feats[-1] = self.bottleneck(feats[-1])

        target_size = x.shape[-2:]
        decoded = []
        for feat, proj in zip(feats, self.proj):
            z = proj(feat)
            if z.shape[-2:] != target_size:
                z = F.interpolate(z, size=target_size, mode="bilinear", align_corners=False)
            decoded.append(z)

        out = self.fuse(torch.cat(decoded, dim=1))

        if valid.shape[1] != out.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand(-1, out.shape[1], -1, -1)
        out = torch.where(valid, out, torch.full_like(out, float("nan")))
        return out


