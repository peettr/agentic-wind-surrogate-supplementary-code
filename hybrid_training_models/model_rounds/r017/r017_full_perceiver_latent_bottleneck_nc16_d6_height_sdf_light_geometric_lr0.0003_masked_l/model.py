import torch
import torch.nn as nn
import torch.nn.functional as F


class _RefConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.conv1 = _RefConv(in_channels, out_channels, 3)
        self.conv2 = _RefConv(out_channels, out_channels, 3)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

    def forward(self, x):
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(y + self.proj(x))


class _Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = _Block(channels, channels)
        self.reduce = nn.Conv2d(channels, channels, 1)
        self.expand = nn.Conv2d(channels, channels, 1)
        self.mix = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x):
        y = self.local(x)
        context = y.mean(dim=(2, 3))
        context = self.mix(context).unsqueeze(-1).unsqueeze(-1)
        return y + self.expand(F.gelu(self.reduce(y) * context))


class perceiver_latent_bottleneck(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_proj = _RefConv(in_channels, channels[0], 3)

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, ch in enumerate(channels):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            self.enc.append(_Block(in_ch, ch))
            if i < depth - 1:
                self.down.append(nn.AvgPool2d(2))

        self.bottleneck = _Bottleneck(channels[-1])

        self.up_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_blocks.append(_Block(channels[i + 1] + channels[i], channels[i]))

        self.out_head = nn.Sequential(
            _Block(channels[0], channels[0]),
            _RefConv(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        valid_out = valid.all(dim=1, keepdim=True)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        y = self.in_proj(x_masked)
        skips = []

        for i, block in enumerate(self.enc):
            y = block(y)
            skips.append(y)
            if i < len(self.down):
                y = self.down[i](y)

        y = self.bottleneck(y)

        for block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))

        y = self.out_head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return torch.where(valid_out, y, torch.full_like(y, float("nan")))