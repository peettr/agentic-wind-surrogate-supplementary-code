import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ReflectConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            ReflectConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, padding=0)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class BoundaryPathMessage(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, padding=0, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
            ReflectConv2d(channels, channels, 3, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        top = x[:, :, :1, :].expand_as(x)
        bottom = x[:, :, -1:, :].expand_as(x)
        left = x[:, :, :, :1].expand_as(x)
        right = x[:, :, :, -1:].expand_as(x)
        msg = 0.25 * (top + bottom + left + right)
        return x + self.fuse(torch.cat([x, msg], dim=1))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class boundary_path_message_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=5):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch

        self.messages = nn.ModuleList([BoundaryPathMessage(ch) for ch in channels])
        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            ReflectConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            h = self.messages[i](h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            h = decoder(h, skip)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != out.shape[1]:
            valid_out = valid_out[:, :1].expand(-1, out.shape[1], -1, -1)

        return torch.where(valid_out, out, torch.full_like(out, float("nan")))


