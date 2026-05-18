import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv(in_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            _ReflectConv(out_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.net(x) + self.skip(x)


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.block = _Block(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.block = _Block(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class asym_lora_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = _Block(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            [_Down(channels[i - 1], channels[i]) for i in range(1, depth)]
        )
        self.bottleneck = _Block(channels[-1], channels[-1])
        self.decoder = nn.ModuleList(
            [
                _Up(channels[i], channels[i - 1], channels[i - 1])
                for i in range(depth - 1, 0, -1)
            ]
        )
        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, padding=0, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for down in self.encoder:
            h = down(h)
            skips.append(h)

        h = self.bottleneck(h)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            h = up(h, skip)

        output = self.head(h)
        output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        output_valid = valid
        if output_valid.shape[1] != output.shape[1]:
            output_valid = output_valid.all(dim=1, keepdim=True)

        output = torch.where(output_valid, output, torch.full_like(output, float("nan")))
        return output


