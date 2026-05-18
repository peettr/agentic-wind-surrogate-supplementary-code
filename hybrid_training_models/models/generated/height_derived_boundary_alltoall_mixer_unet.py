import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for num_groups in (8, 4, 2, 1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, groups=groups, bias=False)
        self.norm = _gn(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(self.pad(x))))


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv(in_channels, out_channels, 3),
            _ReflectConv(out_channels, out_channels, 3),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class _BoundaryAllToAllMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.row = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, 1, bias=False),
        )
        self.col = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, 1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        row_context = self.row(x.mean(dim=3)).unsqueeze(3)
        col_context = self.col(x.mean(dim=2)).unsqueeze(2)
        return x + self.gate(x) * (row_context + col_context)


class height_derived_boundary_alltoall_mixer_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        stem_in = in_channels * 4
        self.stem = _Block(stem_in, channels[0])
        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(_Block(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            _Block(channels[-1], channels[-1]),
            _BoundaryAllToAllMixer(channels[-1]),
            _Block(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(_Block(channels[i + 1] + channels[i], channels[i]))

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        h, w = x_masked.shape[-2:]

        if h >= 2:
            dy_inner = x_masked[:, :, 1:, :] - x_masked[:, :, :-1, :]
            dy = F.pad(dy_inner, (0, 0, 0, 1), mode="reflect")
        else:
            dy = torch.zeros_like(x_masked)

        if w >= 2:
            dx_inner = x_masked[:, :, :, 1:] - x_masked[:, :, :, :-1]
            dx = F.pad(dx_inner, (0, 1, 0, 0), mode="reflect")
        else:
            dx = torch.zeros_like(x_masked)

        valid_f = valid.to(x_masked.dtype)
        x_in = torch.cat([x_masked, dx, dy, valid_f], dim=1)

        skips = []
        y = self.stem(x_in)
        skips.append(y)

        for block in self.encoder:
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))

        y = self.head(y)
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

        valid_any = valid.any(dim=1, keepdim=True)
        y = torch.where(valid_any, y, torch.full_like(y, float("nan")))
        return y