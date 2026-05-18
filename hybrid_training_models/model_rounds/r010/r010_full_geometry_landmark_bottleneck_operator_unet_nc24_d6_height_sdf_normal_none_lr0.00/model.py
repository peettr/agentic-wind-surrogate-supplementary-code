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


class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + r)


class _BottleneckOperator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 8)

        self.local = _ResBlock(channels, channels)

        self.landmark_reduce = nn.Conv2d(channels, hidden, 1)
        self.landmark_expand = nn.Conv2d(hidden, channels, 1)

        self.freq_scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.freq_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.mix = nn.Conv2d(channels * 2, channels, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        local = self.local(x)

        pooled = F.adaptive_avg_pool2d(x, (8, 8))
        pooled = F.silu(self.landmark_reduce(pooled))
        pooled = self.landmark_expand(pooled)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)

        freq = torch.fft.rfft2(x, norm="ortho")
        h_keep = min(12, freq.shape[-2])
        w_keep = min(12, freq.shape[-1])
        low = torch.zeros_like(freq)
        low[:, :, :h_keep, :w_keep] = freq[:, :, :h_keep, :w_keep]
        freq = torch.fft.irfft2(low, s=x.shape[-2:], norm="ortho")
        freq = freq * self.freq_scale + self.freq_bias

        x = local + pooled + freq
        x = self.mix(torch.cat([x, local], dim=1))
        return F.silu(self.norm(x))


class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.block = _ResBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class geometry_landmark_bottleneck_operator_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(_ResBlock(prev, ch))
            prev = ch

        self.down = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            _ResBlock(channels[-1], channels[-1]),
            _BottleneckOperator(channels[-1]),
            _ResBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        current = channels[-1]
        for skip_ch in reversed(channels):
            self.decoders.append(_UpBlock(current, skip_ch, skip_ch))
            current = skip_ch

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.down(h)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoders, reversed(skips)):
            h = dec(h, skip)

        output = self.head(h)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != output.shape[1]:
            out_valid = out_valid.all(dim=1, keepdim=True).expand(-1, output.shape[1], -1, -1)

        output = output.clone()
        output[~out_valid] = float("nan")
        return output


