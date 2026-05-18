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


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            _ReflectConv(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class _SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.silu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = _ConvBlock(in_channels + skip_channels, out_channels)
        self.attn = _SEBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.attn(self.block(x))


class geometry_landmark_operator_token_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels + 3
        for ch in channels:
            self.encoders.append(_ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.bottleneck = nn.Sequential(
            _ConvBlock(channels[-1], channels[-1]),
            _SEBlock(channels[-1]),
            _ConvBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        decoder_in = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.decoders.append(_UpBlock(decoder_in, skip_ch, skip_ch))
            decoder_in = skip_ch

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def _coords(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        rr = torch.sqrt(torch.clamp(xx * xx + yy * yy, min=0.0))
        return torch.cat([xx, yy, rr], dim=1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        original_size = x_masked.shape[-2:]
        x_in = torch.cat([x_masked, self._coords(x_masked)], dim=1)

        skips = []
        h = x_in
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            h = decoder(h, skip)

        out = self.head(h)
        if out.shape[-2:] != original_size:
            out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            out_valid = out_valid.all(dim=1, keepdim=True).expand_as(out)

        out = out.masked_fill(~out_valid, float("nan"))
        return out


