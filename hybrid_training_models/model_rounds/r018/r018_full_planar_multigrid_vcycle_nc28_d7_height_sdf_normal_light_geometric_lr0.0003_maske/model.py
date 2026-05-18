import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(num_channels, max_groups=8):
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=False)
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(self.pad(x))))


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels)
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.skip is not None:
            x = self.skip(x)
        return y + x


class planar_multigrid_vcycle(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.lift = _Block(in_channels, channels[0])
        self.encoder = nn.ModuleList()
        for i in range(depth - 1):
            self.encoder.append(nn.ModuleDict({
                "down": _ReflectConv(channels[i], channels[i + 1], kernel_size=3, stride=2),
                "block": _Block(channels[i + 1], channels[i + 1]),
            }))

        self.bottleneck = nn.Sequential(
            _Block(channels[-1], channels[-1]),
            _Block(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(nn.ModuleDict({
                "fuse": _Block(channels[i + 1] + channels[i], channels[i]),
                "block": _Block(channels[i], channels[i]),
            }))

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.lift(x_masked)
        skips.append(y)

        for level in self.encoder:
            y = level["down"](y)
            y = level["block"](y)
            skips.append(y)

        y = self.bottleneck(y)

        for level, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = level["fuse"](y)
            y = level["block"](y)

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid[:, :1].expand(-1, y.shape[1], -1, -1)
        else:
            valid_out = valid

        y = y.clone()
        y[~valid_out] = torch.nan
        return y


