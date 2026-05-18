import torch
import torch.nn as nn
import torch.nn.functional as F


def _reflect_indices(size, pad_before, pad_after, device):
    if size <= 1:
        return torch.zeros(size + pad_before + pad_after, dtype=torch.long, device=device)

    positions = torch.arange(-pad_before, size + pad_after, device=device)
    period = 2 * size - 2
    indices = positions.remainder(period)
    indices = torch.where(indices >= size, period - indices, indices)
    return indices.long()


def _reflection_pad2d(x, padding):
    if isinstance(padding, int):
        left = right = top = bottom = padding
    elif len(padding) == 2:
        left = right = int(padding[0])
        top = bottom = int(padding[1])
    else:
        left, right, top, bottom = [int(v) for v in padding]

    if left == 0 and right == 0 and top == 0 and bottom == 0:
        return x

    h_idx = _reflect_indices(x.shape[-2], top, bottom, x.device)
    w_idx = _reflect_indices(x.shape[-1], left, right, x.device)
    return x.index_select(-2, h_idx).index_select(-1, w_idx)


class _SafeReflectionPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return _reflection_pad2d(x, self.padding)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            _SafeReflectionPad2d(pad),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels, 3, dilation)
        self.conv2 = _ReflectConv(out_channels, out_channels, 3, 1)
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        residual = x if self.proj is None else self.proj(x)
        return self.conv2(self.conv1(x)) + residual


class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.block = _Block(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class latent_grid_bridge_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.input_block = _Block(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(_Block(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            _Block(channels[-1], channels[-1], dilation=2),
            _Block(channels[-1], channels[-1], dilation=4),
            _Block(channels[-1], channels[-1], dilation=1),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(_UpBlock(channels[i + 1], channels[i], channels[i]))

        self.output_head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, 1),
            _SafeReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        x = self.input_block(x_masked)
        skips.append(x)

        for block in self.encoder:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x = block(x)
            skips.append(x)

        x = self.bottleneck(x)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = block(x, skip)

        x = self.output_head(x)
        x = F.interpolate(x, size=x_masked.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != x.shape[1]:
            valid_out = valid_out[:, :1].expand(-1, x.shape[1], -1, -1)

        return torch.where(valid_out, x, torch.full_like(x, float("nan")))


