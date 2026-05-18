import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class CrossShapeAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.h_pad = nn.ReflectionPad2d((1, 1, 0, 0))
        self.v_pad = nn.ReflectionPad2d((0, 0, 1, 1))
        self.h_conv = nn.Conv2d(channels, channels, kernel_size=(1, 3), groups=1, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=(3, 1), groups=1, bias=False)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.h_conv(self.h_pad(x)) + self.v_conv(self.v_pad(x))
        y = self.mix(y)
        y = self.norm(y)
        return self.act(x + y)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.adapter = CrossShapeAdapter(in_channels)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.adapter(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class cross_shape_decoder_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        self.stem = ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            DownBlock(channels[i - 1], channels[i]) for i in range(1, depth)
        )

        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], channels[-1]),
            CrossShapeAdapter(channels[-1]),
        )

        self.decoder = nn.ModuleList(
            UpBlock(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        )

        self.out_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
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

        y = self.out_head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if y.shape[1] == x.shape[1]:
            out_valid = valid
        else:
            out_valid = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)

        y = y.clone()
        y[~out_valid] = float("nan")
        return y


