import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.skip(x)

class coarse_to_fine_ladder_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = _Block(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(_Block(channels[i - 1], channels[i]))

        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = _Block(channels[-1], channels[-1])

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0, bias=False))
            self.decoder.append(_Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for block in self.encoder:
            h = self.pool(h)
            h = block(h)
            skips.append(h)

        h = self.bottleneck(h)

        for up, block, skip in zip(self.up, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid = valid[:, :1].expand(-1, out.shape[1], -1, -1)

        out = out.clone()
        out[~valid] = float("nan")
        return out