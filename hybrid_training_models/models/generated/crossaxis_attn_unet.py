import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        g1 = min(8, out_channels)
        while out_channels % g1 != 0:
            g1 -= 1
        self.net = nn.Sequential(
            _ReflectConv(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(g1, out_channels),
            nn.GELU(),
            _ReflectConv(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(g1, out_channels),
            nn.GELU(),
        )
        self.skip = nn.Identity() if in_channels == out_channels else _ReflectConv(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.net(x) + self.skip(x)

class _CrossAxisAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.h_proj = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1, bias=True),
        )
        self.w_proj = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 1, bias=True),
        )
        self.mix = _ReflectConv(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h_ctx = x.mean(dim=3)
        w_ctx = x.mean(dim=2)
        h_attn = self.h_proj(h_ctx).sigmoid().unsqueeze(3)
        w_attn = self.w_proj(w_ctx).sigmoid().unsqueeze(2)
        y = self.mix(x * h_attn * w_attn)
        return x + self.gamma * y

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
        self.reduce = _ReflectConv(in_channels, out_channels, 1, bias=False)
        self.block = _Block(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class crossaxis_attn_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.stem = _Block(in_channels, channels[0])

        self.encoder = nn.ModuleList([
            _Down(channels[i - 1], channels[i]) for i in range(1, depth)
        ])

        self.bottleneck = nn.Sequential(
            _Block(channels[-1], channels[-1]),
            _CrossAxisAttention(channels[-1]),
            _Block(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList([
            _Up(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        ])

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GELU(),
            _ReflectConv(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

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

        if valid.shape[1] != y.shape[1]:
            valid = valid.any(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)

        y = y.masked_fill(~valid, float("nan"))
        return y