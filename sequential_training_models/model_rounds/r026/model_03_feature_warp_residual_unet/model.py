import torch
import torch.nn as nn
import torch.nn.functional as F

class feature_warp_residual_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = feature_warp_residual_unet.ReflectConv(in_channels, out_channels, 3, 1, False)
            self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
            self.conv2 = feature_warp_residual_unet.ReflectConv(out_channels, out_channels, 3, 1, False)
            self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        def forward(self, x):
            residual = self.skip(x)
            x = F.silu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return F.silu(x + residual)

    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.down = feature_warp_residual_unet.ReflectConv(in_channels, out_channels, 3, 2, False)
            self.block = feature_warp_residual_unet.ResidualBlock(out_channels, out_channels)

        def forward(self, x):
            return self.block(self.down(x))

    class UpBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.flow = nn.Conv2d(in_channels, 2, 3, padding=1)
            self.reduce = nn.Conv2d(in_channels + skip_channels, out_channels, 1)
            self.block = feature_warp_residual_unet.ResidualBlock(out_channels, out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            flow = torch.tanh(self.flow(x))
            b, _, h, w = flow.shape
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
                torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
                indexing="ij",
            )
            grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, h, w, 2)
            scale = 2.0 / max(float(min(h, w) - 1), 1.0)
            grid = grid + flow.permute(0, 2, 3, 1) * scale
            skip = F.grid_sample(skip, grid, mode="bilinear", padding_mode="reflection", align_corners=True)

            x = torch.cat([x, skip], dim=1)
            x = self.reduce(x)
            return self.block(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.ResidualBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            [self.DownBlock(channels[i - 1], channels[i]) for i in range(1, depth)]
        )

        self.bottleneck = nn.Sequential(
            self.ResidualBlock(channels[-1], channels[-1]),
            self.ResidualBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList(
            [
                self.UpBlock(channels[i], channels[i - 1], channels[i - 1])
                for i in range(depth - 1, 0, -1)
            ]
        )

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3, 1, False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for block in self.encoder:
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = block(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        y = y + x_masked[:, : y.shape[1]]
        y = torch.where(valid[:, : y.shape[1]], y, torch.full_like(y, float("nan")))
        return y


