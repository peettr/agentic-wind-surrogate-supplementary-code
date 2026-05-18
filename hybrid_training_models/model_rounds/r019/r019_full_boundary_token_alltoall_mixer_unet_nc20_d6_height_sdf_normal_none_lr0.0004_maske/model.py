import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for groups in range(min(8, num_channels), 0, -1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv(in_channels, out_channels, 3),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            _ReflectConv(out_channels, out_channels, 3),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Identity() if in_channels == out_channels else _ReflectConv(in_channels, out_channels, 1)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = _ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return self.block(x)


class _Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = _ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class _BoundaryTokenMixer(nn.Module):
    def __init__(self, channels, tokens=80, bottleneck=4):
        super().__init__()
        hidden = max(channels // bottleneck, 8)
        self.tokens = tokens
        self.q = nn.Linear(channels, hidden, bias=False)
        self.k = nn.Linear(channels, hidden, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.scale = hidden ** -0.5
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        t = min(self.tokens, h, w)

        top = F.adaptive_avg_pool1d(x[:, :, 0, :], t)
        bottom = F.adaptive_avg_pool1d(x[:, :, -1, :], t)
        left = F.adaptive_avg_pool1d(x[:, :, :, 0], t)
        right = F.adaptive_avg_pool1d(x[:, :, :, -1], t)

        tokens = torch.cat([top, bottom, left, right], dim=-1).transpose(1, 2)
        tokens = self.norm(tokens)

        q = self.q(tokens)
        k = self.k(tokens)
        v = self.v(tokens)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        mixed = self.proj(torch.matmul(attn, v)).mean(dim=1).view(b, c, 1, 1)

        return x + self.gate * mixed


class boundary_token_alltoall_mixer_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = _ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            [_Down(channels[i - 1], channels[i]) for i in range(1, depth)]
        )

        self.bottleneck = nn.Sequential(
            _ConvBlock(channels[-1], channels[-1]),
            _BoundaryTokenMixer(channels[-1]),
            _ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList(
            [_Up(channels[i], channels[i - 1], channels[i - 1]) for i in range(depth - 1, 0, -1)]
        )

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            _ReflectConv(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for down in self.encoder:
            y = down(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            y = up(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if y.shape[1] == x.shape[1]:
            out_valid = valid
        else:
            out_valid = valid[:, :1].expand(-1, y.shape[1], -1, -1)

        y = torch.where(out_valid, y, torch.full_like(y, float("nan")))
        return y