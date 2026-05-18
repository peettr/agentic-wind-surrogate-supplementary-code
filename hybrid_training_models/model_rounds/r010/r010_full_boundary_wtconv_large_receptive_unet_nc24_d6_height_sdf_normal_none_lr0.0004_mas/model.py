import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    groups = min(8, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _reflection_indices(size, padding, device):
    if size <= 1:
        return torch.zeros(size + 2 * padding, dtype=torch.long, device=device)

    period = 2 * size - 2
    idx = torch.arange(-padding, size + padding, device=device)
    idx = torch.remainder(idx, period)
    idx = torch.minimum(idx, period - idx)
    return idx.long()


def _reflect_pad2d(x, padding):
    if padding == 0:
        return x

    height, width = x.shape[-2:]
    rows = _reflection_indices(height, padding, x.device)
    cols = _reflection_indices(width, padding, x.device)
    return x.index_select(-2, rows).index_select(-1, cols)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1, bias=False):
        super().__init__()
        self.padding = kernel_size // 2
        self.pad = nn.ReflectionPad2d(self.padding) if self.padding > 0 else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        if self.padding > 0:
            if x.shape[-2] > self.padding and x.shape[-1] > self.padding:
                x = self.pad(x)
            else:
                x = _reflect_pad2d(x, self.padding)
        return self.conv(x)


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = _ReflectConv(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

        self.conv1 = _ReflectConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = _gn(out_channels)

        self.dw_large = _ReflectConv(out_channels, out_channels, 9, groups=out_channels, bias=False)
        self.pw_large = _ReflectConv(out_channels, out_channels, 1, bias=False)
        self.norm2 = _gn(out_channels)

        self.conv2 = _ReflectConv(out_channels, out_channels, 3, bias=False)
        self.norm3 = _gn(out_channels)

    def forward(self, x):
        residual = self.proj(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.gelu(x)

        x = self.dw_large(x)
        x = self.pw_large(x)
        x = self.norm2(x)
        x = F.gelu(x)

        x = self.conv2(x)
        x = self.norm3(x)

        return F.gelu(x + residual)


class boundary_wtconv_large_receptive_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.depth = depth
        self.channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = _Block(in_channels, self.channels[0])

        self.encoder = nn.ModuleList()
        for i in range(depth):
            in_ch = self.channels[i - 1] if i > 0 else self.channels[0]
            out_ch = self.channels[i]
            self.encoder.append(_Block(in_ch, out_ch))

        self.bottleneck = nn.Sequential(
            _Block(self.channels[-1], self.channels[-1]),
            _Block(self.channels[-1], self.channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(_Block(self.channels[i + 1] + self.channels[i], self.channels[i]))

        self.head = nn.Sequential(
            _Block(self.channels[0], self.channels[0]),
            _ReflectConv(self.channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        x = self.stem(x_masked)
        skips = []

        for i, block in enumerate(self.encoder):
            x = block(x)
            skips.append(x)
            if i < self.depth - 1 and min(x.shape[-2:]) > 1:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        output = self.head(x)
        output = F.interpolate(output, size=x_masked.shape[-2:], mode="bilinear", align_corners=False)

        output_mask = valid
        if output_mask.shape[1] != output.shape[1]:
            output_mask = output_mask.all(dim=1, keepdim=True)

        output = torch.where(output_mask, output, torch.full_like(output, float("nan")))
        return output