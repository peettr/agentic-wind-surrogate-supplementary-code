import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ResidualMambaLikeBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.in_proj = ReflectionConv2d(channels, channels * 2, 1)
        self.dw_h = nn.Sequential(
            nn.ReflectionPad2d((3, 3, 0, 0)),
            nn.Conv2d(channels, channels, (1, 7), groups=channels, padding=0),
        )
        self.dw_w = nn.Sequential(
            nn.ReflectionPad2d((0, 0, 3, 3)),
            nn.Conv2d(channels, channels, (7, 1), groups=channels, padding=0),
        )
        self.out_proj = ReflectionConv2d(channels, channels, 1)

        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            ReflectionConv2d(channels, channels * 4, 1),
            nn.GELU(),
            ReflectionConv2d(channels * 4, channels, 1),
        )

    def forward(self, x):
        y = self.norm1(x)
        y, gate = self.in_proj(y).chunk(2, dim=1)
        y = self.dw_h(y) + self.dw_w(y)
        y = self.out_proj(y * torch.sigmoid(gate))
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = ReflectionConv2d(in_channels, out_channels, 3)
        self.block = ResidualMambaLikeBlock(out_channels)

    def forward(self, x):
        x = self.proj(x)
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.proj = ReflectionConv2d(in_channels + skip_channels, out_channels, 3)
        self.block = ResidualMambaLikeBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        x = self.block(x)
        return x


class encoder_mamba_residual_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = DownBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(DownBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            ResidualMambaLikeBlock(channels[-1]),
            ResidualMambaLikeBlock(channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3),
            nn.GELU(),
            ReflectionConv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for block in self.encoder:
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = block(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            if valid.shape[1] == 1:
                valid = valid.expand(-1, y.shape[1], -1, -1)
            elif y.shape[1] == 1:
                valid = valid.all(dim=1, keepdim=True)
            else:
                valid = valid[:, : y.shape[1]]

        y = y.masked_fill(~valid, float("nan"))
        return y


