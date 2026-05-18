import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


def _group_count(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = _ReflectConv(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _SeparableConv(in_channels, out_channels)
        self.conv2 = _SeparableConv(out_channels, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class latent_grid_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        if n_c < 1:
            raise ValueError("n_c must be >= 1")
        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()

        prev_channels = in_channels
        for i, ch in enumerate(channels):
            self.encoder.append(_ConvBlock(prev_channels, ch))
            if i < depth - 1:
                self.down.append(_SeparableConv(ch, channels[i + 1], stride=2))
            prev_channels = channels[i + 1] if i < depth - 1 else ch

        self.bottleneck = _ConvBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        current_channels = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            self.decoder.append(_ConvBlock(current_channels + skip_channels, skip_channels))
            current_channels = skip_channels

        self.head = nn.Conv2d(current_channels, out_channels, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        nan_mask = torch.isnan(x)
        valid = ~nan_mask.any(dim=1, keepdim=True)
        x = torch.where(nan_mask, torch.zeros_like(x), x)

        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            skips.append(x)
            if i < len(self.down):
                x = self.down[i](x)

        x = self.bottleneck(x)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((x, skip), dim=1)
            x = block(x)

        x = self.head(x)

        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        valid = valid.expand(-1, x.shape[1], -1, -1)
        x = x.clone()
        x[~valid] = float("nan")
        return x


