import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, bias=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class GatedDWBlock(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1

        self.norm1 = nn.GroupNorm(groups, channels)
        self.dw = ReflectConv2d(channels, channels, 3, groups=channels)
        self.pw1 = nn.Conv2d(channels, hidden * 2, 1)
        self.pw2 = nn.Conv2d(hidden, channels, 1)

        self.norm2 = nn.GroupNorm(groups, channels)
        self.ffn1 = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn_dw = ReflectConv2d(hidden * 2, hidden * 2, 3, groups=hidden * 2)
        self.ffn2 = nn.Conv2d(hidden, channels, 1)

        self.scale1 = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.scale2 = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm1(x)
        y = self.dw(y)
        a, b = self.pw1(y).chunk(2, dim=1)
        y = self.pw2(F.gelu(a) * b)
        x = x + self.scale1 * y

        y = self.norm2(x)
        a, b = self.ffn_dw(self.ffn1(y)).chunk(2, dim=1)
        y = self.ffn2(F.gelu(a) * b)
        return x + self.scale2 * y

class Stage(nn.Module):
    def __init__(self, channels, num_blocks=2):
        super().__init__()
        self.blocks = nn.Sequential(*[GatedDWBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        return self.blocks(x)

class restormer_refine_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        depth = max(1, int(depth))
        self.depth = depth
        self.channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = ReflectConv2d(in_channels, self.channels[0], 3)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, ch in enumerate(self.channels):
            self.encoders.append(Stage(ch, num_blocks=2))
            if i < depth - 1:
                self.downs.append(nn.Sequential(
                    nn.AvgPool2d(2),
                    ReflectConv2d(ch, self.channels[i + 1], 3)
                ))

        self.bottleneck = Stage(self.channels[-1], num_blocks=3)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.ups.append(nn.Conv2d(self.channels[i + 1], self.channels[i], 1))
            self.decoders.append(nn.Sequential(
                ReflectConv2d(self.channels[i] * 2, self.channels[i], 3),
                Stage(self.channels[i], num_blocks=2)
            ))

        self.refine = nn.Sequential(
            Stage(self.channels[0], num_blocks=2),
            ReflectConv2d(self.channels[0], self.channels[0], 3),
            nn.GELU(),
            nn.Conv2d(self.channels[0], out_channels, 1)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        y = self.stem(x_masked)
        skips = []

        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.downs[i](y)

        y = self.bottleneck(y)

        for i in range(self.depth - 2, -1, -1):
            y = F.interpolate(y, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            y = self.ups[self.depth - 2 - i](y)
            y = torch.cat([y, skips[i]], dim=1)
            y = self.decoders[self.depth - 2 - i](y)

        y = self.refine(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return torch.where(valid, y, torch.full_like(y, float("nan")))


