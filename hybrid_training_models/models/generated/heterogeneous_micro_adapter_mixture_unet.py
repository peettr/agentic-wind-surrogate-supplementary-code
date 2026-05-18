import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ReflectedSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, activation=True):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.pad(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.activation(x)


class _MicroAdapterExpert(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.pad = nn.ReflectionPad2d(pad)
        self.spatial = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=hidden_channels,
            bias=False,
        )
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.reduce(x)
        x = self.pad(x)
        x = self.spatial(x)
        x = self.expand(x)
        x = self.norm(x)
        return self.activation(x)


class _MicroAdapterMixture(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_channels = max(1, channels // 4)
        self.experts = nn.ModuleList(
            [
                _MicroAdapterExpert(channels, hidden_channels, kernel_size=3, dilation=1),
                _MicroAdapterExpert(channels, hidden_channels, kernel_size=3, dilation=2),
                _MicroAdapterExpert(channels, hidden_channels, kernel_size=5, dilation=1),
            ]
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, len(self.experts), kernel_size=1),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        weights = torch.softmax(self.gate(x), dim=1)
        mixed = torch.zeros_like(x)
        for idx, expert in enumerate(self.experts):
            mixed = mixed + expert(x) * weights[:, idx:idx + 1]
        return x + self.scale * mixed


class _ResidualAdapterBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectedSeparableConv(in_channels, out_channels, kernel_size=3)
        self.conv2 = _ReflectedSeparableConv(out_channels, out_channels, kernel_size=3, activation=False)
        self.adapter = _MicroAdapterMixture(out_channels)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(_num_groups(out_channels), out_channels),
            )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.adapter(x)
        return self.activation(x + residual)


class _UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.block = _ResidualAdapterBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class heterogeneous_micro_adapter_mixture_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        if n_c <= 0:
            raise ValueError("n_c must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        channels = [min(n_c * (2 ** level), n_c * 8) for level in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(_ResidualAdapterBlock(prev_channels, ch))
            prev_channels = ch

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = _ResidualAdapterBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        for level in range(depth - 2, -1, -1):
            self.decoders.append(_UpBlock(channels[level + 1], channels[level], channels[level]))

        self.output = nn.Conv2d(channels[0], out_channels, kernel_size=1)

        total_params = sum(param.numel() for param in self.parameters())
        if total_params >= 50_000_000:
            raise ValueError("model has 50M or more parameters; reduce n_c or depth")

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked
        for idx, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if idx != len(self.encoders) - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            y = decoder(y, skip)

        output = self.output(y)

        if valid.shape != output.shape:
            if valid.shape[1] == 1:
                valid = valid.expand_as(output)
            else:
                valid = valid.all(dim=1, keepdim=True).expand_as(output)

        output = output.clone()
        output[~valid] = float("nan")
        return output