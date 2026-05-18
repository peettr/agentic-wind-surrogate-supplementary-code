import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels, max_groups=8):
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class coordinate_moe_implicit_residual_head_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = coordinate_moe_implicit_residual_head_unet.ReflectConv(channels, channels, 3)
            self.norm1 = _gn(channels)
            self.conv2 = coordinate_moe_implicit_residual_head_unet.ReflectConv(channels, channels, 3)
            self.norm2 = _gn(channels)

        def forward(self, x):
            r = x
            x = F.silu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return F.silu(x + r)

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.proj = coordinate_moe_implicit_residual_head_unet.ReflectConv(in_channels, out_channels, 3)
            self.norm = _gn(out_channels)
            self.res = coordinate_moe_implicit_residual_head_unet.ResBlock(out_channels)

        def forward(self, x):
            x = F.silu(self.norm(self.proj(x)))
            return self.res(x)

    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.pool = nn.AvgPool2d(2)
            self.block = coordinate_moe_implicit_residual_head_unet.ConvBlock(in_channels, out_channels)

        def forward(self, x):
            return self.block(self.pool(x))

    class UpBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.block = coordinate_moe_implicit_residual_head_unet.ConvBlock(in_channels + skip_channels, out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return self.block(x)

    class CoordinateMoEImplicitResidualHead(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, experts=4):
            super().__init__()
            self.experts = nn.ModuleList([
                nn.Sequential(
                    coordinate_moe_implicit_residual_head_unet.ReflectConv(in_channels + 2, hidden_channels, 3),
                    _gn(hidden_channels),
                    nn.SiLU(),
                    coordinate_moe_implicit_residual_head_unet.ReflectConv(hidden_channels, hidden_channels, 3),
                    _gn(hidden_channels),
                    nn.SiLU(),
                    coordinate_moe_implicit_residual_head_unet.ReflectConv(hidden_channels, out_channels, 3),
                )
                for _ in range(experts)
            ])
            self.gate = nn.Sequential(
                coordinate_moe_implicit_residual_head_unet.ReflectConv(in_channels + 2, hidden_channels, 3),
                _gn(hidden_channels),
                nn.SiLU(),
                coordinate_moe_implicit_residual_head_unet.ReflectConv(hidden_channels, experts, 1),
            )
            self.residual = coordinate_moe_implicit_residual_head_unet.ReflectConv(in_channels, out_channels, 1)

        def forward(self, x):
            b, _, h, w = x.shape
            yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
            xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
            z = torch.cat([x, xx, yy], dim=1)
            weights = torch.softmax(self.gate(z), dim=1)
            y = 0.0
            for i, expert in enumerate(self.experts):
                y = y + weights[:, i:i + 1] * expert(z)
            return y + self.residual(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [n_c * min(2 ** i, 8) for i in range(depth)]

        self.stem = self.ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList([
            self.DownBlock(channels[i - 1], channels[i])
            for i in range(1, depth)
        ])

        self.bottleneck = nn.Sequential(
            self.ResBlock(channels[-1]),
            self.ResBlock(channels[-1])
        )

        self.decoder = nn.ModuleList([
            self.UpBlock(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        ])

        self.head = self.CoordinateMoEImplicitResidualHead(
            channels[0],
            max(n_c, channels[0]),
            out_channels,
            experts=4
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

        y = self.head(y)

        if valid.shape[1] != y.shape[1]:
            valid = valid[:, :1].expand_as(y)
        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y