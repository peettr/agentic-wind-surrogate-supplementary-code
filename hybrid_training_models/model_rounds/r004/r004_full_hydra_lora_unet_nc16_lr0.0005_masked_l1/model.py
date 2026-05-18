import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectionConv2d(in_channels, out_channels, 3),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            _ReflectionConv2d(out_channels, out_channels, 3),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Identity() if in_channels == out_channels else _ReflectionConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.net(x) + self.skip(x)

class _Bottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 4, 8)
        self.local = _ConvBlock(channels, channels)
        self.lora_down = _ReflectionConv2d(channels, mid, 1, bias=False)
        self.lora_up = _ReflectionConv2d(mid, channels, 1, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.local(x) + self.gate * self.lora_up(F.silu(self.lora_down(x)))

class hydra_lora_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_proj = _ConvBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(depth - 1):
            self.down.append(_ReflectionConv2d(channels[i], channels[i + 1], 3, stride=2))
            self.encoder.append(_ConvBlock(channels[i + 1], channels[i + 1]))

        self.bottleneck = _Bottleneck(channels[-1])

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up.append(_ReflectionConv2d(channels[i + 1], channels[i], 1))
            self.decoder.append(_ConvBlock(channels[i] * 2, channels[i]))

        self.out_head = nn.Sequential(
            _ReflectionConv2d(channels[0], channels[0], 3),
            nn.SiLU(inplace=True),
            _ReflectionConv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = self.in_proj(x_masked)
        skips.append(h)

        for down, enc in zip(self.down, self.encoder):
            h = enc(down(h))
            skips.append(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1))

        output = self.out_head(h)
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid = valid.expand_as(output)
        output = output.clone()
        output[~valid] = float("nan")
        return output