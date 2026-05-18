import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, bias=False):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class SelfGatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = None
        if in_channels != out_channels:
            self.proj = ReflectionConv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.conv1 = ReflectionConv2d(in_channels, out_channels, kernel_size=3, bias=False)
        self.norm1 = nn.GroupNorm(self._groups(out_channels), out_channels)

        self.conv2 = ReflectionConv2d(out_channels, out_channels, kernel_size=3, bias=False)
        self.norm2 = nn.GroupNorm(self._groups(out_channels), out_channels)

        self.gate = nn.Sequential(
            ReflectionConv2d(out_channels, out_channels, kernel_size=3, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def _groups(channels):
        for g in (8, 6, 4, 3, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def forward(self, x):
        residual = x if self.proj is None else self.proj(x)

        y = self.conv1(x)
        y = self.norm1(y)
        y = F.silu(y, inplace=False)

        y = self.conv2(y)
        y = self.norm2(y)

        y = y * self.gate(y)
        return F.silu(y + residual, inplace=False)


class BoundaryRefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            ReflectionConv2d(channels + 1, channels, kernel_size=3, bias=False),
            nn.GroupNorm(SelfGatedBlock._groups(channels), channels),
            nn.SiLU(inplace=False),
            ReflectionConv2d(channels, channels, kernel_size=3, bias=True),
        )

    def forward(self, features, source):
        dx = source[:, :, :, 1:] - source[:, :, :, :-1]
        dy = source[:, :, 1:, :] - source[:, :, :-1, :]

        dx = F.pad(dx.abs(), (0, 1, 0, 0), mode="replicate")
        dy = F.pad(dy.abs(), (0, 0, 0, 1), mode="replicate")

        boundary = dx + dy
        boundary = boundary / (boundary.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        boundary = F.interpolate(boundary, size=features.shape[-2:], mode="bilinear", align_corners=False)

        return features + self.refine(torch.cat([features, boundary], dim=1))


class self_gated_boundary_refine_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = SelfGatedBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.encoder.append(SelfGatedBlock(prev, ch))
            self.down.append(nn.AvgPool2d(kernel_size=2, stride=2))
            prev = ch

        self.bottleneck = nn.Sequential(
            SelfGatedBlock(channels[-1], channels[-1]),
            SelfGatedBlock(channels[-1], channels[-1]),
        )

        self.up_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for ch in reversed(channels):
            self.up_proj.append(ReflectionConv2d(prev, ch, kernel_size=1, bias=False))
            self.decoder.append(SelfGatedBlock(ch * 2, ch))
            prev = ch

        self.boundary_refine = BoundaryRefineBlock(channels[0])

        self.head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], kernel_size=3, bias=False),
            nn.GroupNorm(SelfGatedBlock._groups(channels[0]), channels[0]),
            nn.SiLU(inplace=False),
            ReflectionConv2d(channels[0], out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        source = x_masked
        y = self.stem(x_masked)

        skips = []
        for block, down in zip(self.encoder, self.down):
            y = block(y)
            skips.append(y)
            y = down(y)

        y = self.bottleneck(y)

        for proj, block, skip in zip(self.up_proj, self.decoder, reversed(skips)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = proj(y)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        y = self.boundary_refine(y, source)
        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid[:, :1].expand(-1, y.shape[1], -1, -1)

        y = y.clone()
        y[~valid] = torch.nan
        return y