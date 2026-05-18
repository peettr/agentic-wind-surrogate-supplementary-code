import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size
        self.pad = nn.ReflectionPad2d((kw // 2, kw // 2, kh // 2, kh // 2))
        self.conv = nn.Conv2d(in_channels, out_channels, (kh, kw), padding=0, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class _Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.v = _ReflectConv(channels, channels, 3)
        self.g = _ReflectConv(channels, channels, 3)
        self.ax = _ReflectConv(channels, channels, (1, 7), groups=channels)
        self.ay = _ReflectConv(channels, channels, (7, 1), groups=channels)
        self.mix = nn.Conv2d(channels, channels, 1)
        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x):
        h = self.norm1(x)
        h = self.v(h) * torch.sigmoid(self.g(h))
        h = self.mix(self.ax(h) + self.ay(h))
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x

class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels * 4, out_channels, 1)
        self.block = _Block(out_channels)

    def forward(self, x):
        x = F.pixel_unshuffle(x, 2)
        x = self.proj(x)
        return self.block(x)

class _Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels + skip_channels, out_channels * 4, 1)
        self.block = _Block(out_channels)

    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = F.pixel_shuffle(self.proj(x), 2)
        return self.block(x)

class wavelet_gated_anisotropic(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = nn.Sequential(
            _ReflectConv(in_channels, channels[0], 3),
            _Block(channels[0]),
        )

        self.encoder = nn.ModuleList(
            [_Down(channels[i], channels[i + 1]) for i in range(depth - 1)]
        )

        self.bottleneck = nn.Sequential(
            _Block(channels[-1]),
            _Block(channels[-1]),
        )

        self.decoder = nn.ModuleList(
            [
                _Up(channels[i + 1], channels[i], channels[i])
                for i in range(depth - 2, -1, -1)
            ]
        )

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3),
            nn.GELU(),
            _ReflectConv(channels[0], out_channels, 3),
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

        skips = skips[:-1][::-1]
        for up, skip in zip(self.decoder, skips):
            h = up(h, skip)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            out_valid = out_valid[:, :1].expand(-1, out.shape[1], -1, -1)
        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out


