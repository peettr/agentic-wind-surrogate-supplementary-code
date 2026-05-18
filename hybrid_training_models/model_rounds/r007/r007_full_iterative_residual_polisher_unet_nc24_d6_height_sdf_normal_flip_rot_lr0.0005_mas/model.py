import torch
import torch.nn as nn
import torch.nn.functional as F


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv(channels, channels, 3, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
            _ReflectConv(channels, channels, 3, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class _DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = _ReflectConv(in_channels, out_channels, 3, stride=2, bias=False)
        self.net = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            _ResBlock(out_channels),
        )

    def forward(self, x):
        return self.net(self.down(x))


class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            _ReflectConv(in_channels + skip_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            _ResBlock(out_channels),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class iterative_residual_polisher_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.stem = nn.Sequential(
            _ReflectConv(in_channels, channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            _ResBlock(channels[0]),
        )

        self.encoder = nn.ModuleList(
            _DownBlock(channels[i], channels[i + 1]) for i in range(depth - 1)
        )

        self.bottleneck = nn.Sequential(
            _ResBlock(channels[-1]),
            _ResBlock(channels[-1]),
        )

        self.decoder = nn.ModuleList(
            _UpBlock(channels[i + 1], channels[i], channels[i]) for i in range(depth - 2, -1, -1)
        )

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

        self.polisher = nn.Sequential(
            _ReflectConv(out_channels + in_channels, channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            _ResBlock(channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for block in self.encoder:
            h = block(h)
            skips.append(h)

        h = self.bottleneck(h)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = block(h, skip)

        residual = self.head(h)
        output = residual

        polish = self.polisher(torch.cat([x_masked, output], dim=1))
        output = output + polish

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != output.shape[1]:
            valid_out = valid_out.all(dim=1, keepdim=True)
            if valid_out.shape[1] != output.shape[1]:
                valid_out = valid_out.expand(-1, output.shape[1], -1, -1)

        output = torch.where(valid_out, output, torch.full_like(output, float("nan")))
        return output