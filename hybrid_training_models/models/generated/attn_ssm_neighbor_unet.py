import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(y + self.skip(x))

class _NeighborSSM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw_h = nn.Conv2d(channels, channels, (1, 7), padding=0, groups=channels, bias=False)
        self.dw_v = nn.Conv2d(channels, channels, (7, 1), padding=0, groups=channels, bias=False)
        self.pad_h = nn.ReflectionPad2d((3, 3, 0, 0))
        self.pad_v = nn.ReflectionPad2d((0, 0, 3, 3))
        self.gate = nn.Conv2d(channels, channels, 1, padding=0)
        self.proj = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        h = self.dw_h(self.pad_h(x))
        v = self.dw_v(self.pad_v(x))
        g = torch.sigmoid(self.gate(x))
        y = self.proj((h + v) * g)
        return F.gelu(self.norm(y + x))

class _AttentionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 1)
        self.q = nn.Conv2d(channels, hidden, 1, padding=0, bias=False)
        self.k = nn.Conv2d(channels, hidden, 1, padding=0, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, padding=0, bias=False)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / (q.shape[1] ** 0.5))
        return self.proj(v * attn) + x

class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = _Block(in_channels, out_channels)
        self.ssm = _NeighborSSM(out_channels)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = self.block(x)
        return self.ssm(x)

class _Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.gate = _AttentionGate(skip_channels)
        self.block = _Block(out_channels + skip_channels, out_channels)
        self.ssm = _NeighborSSM(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        skip = self.gate(skip)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return self.ssm(x)

class attn_ssm_neighbor_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=5):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.channels = channels

        self.stem = _Block(in_channels, channels[0])
        self.encoders = nn.ModuleList([
            _Down(channels[i - 1], channels[i]) for i in range(1, depth)
        ])

        self.bottleneck = nn.Sequential(
            _Block(channels[-1], channels[-1]),
            _NeighborSSM(channels[-1]),
            _AttentionGate(channels[-1]),
            _Block(channels[-1], channels[-1])
        )

        self.decoders = nn.ModuleList([
            _Up(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        ])

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 1, padding=0)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for encoder in self.encoders:
            y = encoder(y)
            skips.append(y)

        y = self.bottleneck(y)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            y = decoder(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)

        output = y.clone()
        output[~valid] = float("nan")
        return output