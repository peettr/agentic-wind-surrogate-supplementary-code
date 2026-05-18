import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, bias=False):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.pad_size = pad
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def _reflect_pad(self, x, pad):
        if pad == 0:
            return x

        remaining = pad
        while remaining > 0:
            h, w = x.shape[-2:]
            step_h = min(remaining, max(h - 1, 0))
            step_w = min(remaining, max(w - 1, 0))

            if step_h == 0 or step_w == 0:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                continue

            step = min(step_h, step_w)
            x = F.pad(x, (step, step, step, step), mode="reflect")
            remaining -= step

        return x

    def forward(self, x):
        return self.conv(self._reflect_pad(x, self.pad_size))


class DilatedModulationBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = ReflectionConv2d(channels, channels, 3, dilation=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = ReflectionConv2d(channels, channels, 3, dilation=2)
        self.norm3 = nn.GroupNorm(min(8, channels), channels)
        self.conv3 = ReflectionConv2d(channels, channels, 3, dilation=4)

        hidden = max(channels // 4, 8)
        self.modulation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        residual = x
        y = self.conv1(F.silu(self.norm1(x), inplace=True))
        y = self.conv2(F.silu(self.norm2(y), inplace=True))
        y = self.conv3(F.silu(self.norm3(y), inplace=True))
        y = y * self.modulation(y)
        return residual + y


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = ReflectionConv2d(in_channels, out_channels, 3)
        self.block1 = DilatedModulationBlock(out_channels)
        self.block2 = DilatedModulationBlock(out_channels)

    def forward(self, x):
        x = self.proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = ReflectionConv2d(in_channels + skip_channels, out_channels, 3)
        self.block1 = DilatedModulationBlock(out_channels)
        self.block2 = DilatedModulationBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class dilated_modulation_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_adapter = ReflectionConv2d(in_channels, channels[0], 3)

        self.encoder = nn.ModuleList()
        prev_channels = channels[0]
        for ch in channels:
            self.encoder.append(EncoderStage(prev_channels, ch))
            prev_channels = ch

        self.bottleneck = nn.Sequential(
            DilatedModulationBlock(channels[-1]),
            DilatedModulationBlock(channels[-1]),
            DilatedModulationBlock(channels[-1]),
        )

        self.decoder = nn.ModuleList()
        current_channels = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            self.decoder.append(DecoderStage(current_channels, skip_channels, skip_channels))
            current_channels = skip_channels

        self.output_head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            ReflectionConv2d(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        h = self.input_adapter(x_masked)
        skips = []

        for i, stage in enumerate(self.encoder):
            h = stage(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for stage, skip in zip(self.decoder, reversed(skips[:-1])):
            h = stage(h, skip)

        out = self.output_head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid = valid.any(dim=1, keepdim=True).expand_as(out)

        out = out.masked_fill(~valid, float("nan"))
        return out


