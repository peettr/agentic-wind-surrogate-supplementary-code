import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.conv1 = _ReflectConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        residual = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)

class coord_residual_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels + 2
        for ch in channels:
            self.encoders.append(_ResBlock(prev_channels, ch))
            prev_channels = ch

        self.bottleneck = _ResBlock(channels[-1], channels[-1])

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(channels[i + 1], channels[i], 1))
            self.decoders.append(_ResBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def _coords(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        yy = yy.expand(b, 1, h, w)
        xx = xx.expand(b, 1, h, w)
        return torch.cat((xx, yy), dim=1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        y = torch.cat((x_masked, self._coords(x_masked)), dim=1)

        skips = []
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i < len(self.encoders) - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_conv(y)
            y = decoder(torch.cat((y, skip), dim=1))

        y = self.head(y)
        y = y + x_masked[:, :y.shape[1]]

        return torch.where(valid[:, :y.shape[1]], y, torch.full_like(y, float("nan")))


