import torch
import torch.nn as nn
import torch.nn.functional as F

def _num_groups(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1

class _ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))

class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv2d(in_channels, out_channels, kernel_size=3, bias=False)
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = _ReflectConv2d(out_channels, out_channels, kernel_size=3, bias=False)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)

class planar_multigrid_vcycle_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be at least 1")
        if n_c < 1:
            raise ValueError("n_c must be at least 1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        enc_in = in_channels + 1
        for enc_out in self.channels:
            self.encoder.append(_ResidualBlock(enc_in, enc_out))
            enc_in = enc_out

        self.bottleneck = nn.Sequential(
            _ResidualBlock(self.channels[-1], self.channels[-1]),
            _ResidualBlock(self.channels[-1], self.channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(
                _ResidualBlock(self.channels[i + 1] + self.channels[i], self.channels[i])
            )

        self.head = nn.Conv2d(
            self.channels[0],
            out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

        if sum(p.numel() for p in self.parameters()) >= 50_000_000:
            raise ValueError("model has 50M or more parameters; reduce n_c or depth")

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        pixel_valid = valid.all(dim=1, keepdim=True)

        h = torch.cat([x_masked, pixel_valid.to(dtype=x_masked.dtype)], dim=1)

        skips = []
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i != self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] == output.shape[1]:
            valid_out = valid
        else:
            valid_out = pixel_valid.expand(-1, output.shape[1], -1, -1)

        output = output.clone()
        output[~valid_out] = float("nan")
        return output