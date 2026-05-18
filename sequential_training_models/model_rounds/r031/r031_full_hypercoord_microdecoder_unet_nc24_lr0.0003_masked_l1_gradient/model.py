import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            _ReflectConv(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.net(x) + self.skip(x)

class hypercoord_microdecoder_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.coord_proj = nn.Sequential(
            nn.Conv2d(2, n_c, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(n_c, n_c, 1),
        )

        self.input_proj = _ReflectConv(in_channels, channels[0], 3, bias=False)

        self.encoder = nn.ModuleList()
        prev = channels[0] + n_c
        for ch in channels:
            self.encoder.append(_Block(prev, ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            _Block(channels[-1], channels[-1]),
            _Block(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(_Block(channels[i + 1] + channels[i], channels[i]))

        self.microdecoder = nn.Sequential(
            _ReflectConv(channels[0] + n_c, channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def _coords(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat((xx, yy), dim=1)

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        coords = self._coords(x_masked)
        coord_features = self.coord_proj(coords)

        h = self.input_proj(x_masked)
        h = torch.cat((h, coord_features), dim=1)

        skips = []
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat((h, skip), dim=1)
            h = block(h)

        h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat((h, coord_features), dim=1)
        out = self.microdecoder(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = out.masked_fill(~valid.expand(-1, self.out_channels, -1, -1), float("nan"))
        return out