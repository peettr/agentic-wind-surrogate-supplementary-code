import torch
import torch.nn as nn
import torch.nn.functional as F


class strip_mixer_refine_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
            super().__init__()
            if isinstance(kernel_size, tuple):
                pad_h = kernel_size[0] // 2
                pad_w = kernel_size[1] // 2
            else:
                pad_h = pad_w = kernel_size // 2
            self.pad = (pad_w, pad_w, pad_h, pad_h)
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=0,
                groups=groups,
                bias=False,
            )

        @staticmethod
        def _reflect_pad_chunked(x, pad):
            left, right, top, bottom = pad
            while left or right or top or bottom:
                h, w = x.shape[-2:]
                step_left = min(left, max(w - 1, 0))
                step_right = min(right, max(w - 1, 0))
                step_top = min(top, max(h - 1, 0))
                step_bottom = min(bottom, max(h - 1, 0))

                if step_left == step_right == step_top == step_bottom == 0:
                    raise RuntimeError("reflection padding requires spatial dimensions greater than 1")

                x = F.pad(
                    x,
                    (step_left, step_right, step_top, step_bottom),
                    mode="reflect",
                )
                left -= step_left
                right -= step_right
                top -= step_top
                bottom -= step_bottom
            return x

        def forward(self, x):
            return self.conv(self._reflect_pad_chunked(x, self.pad))

    class StripMixerBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            groups = 8 if channels % 8 == 0 else 4 if channels % 4 == 0 else 1
            self.norm1 = nn.GroupNorm(groups, channels)
            self.vstrip = strip_mixer_refine_unet.ReflectConv(channels, channels, (9, 1), groups=channels)
            self.hstrip = strip_mixer_refine_unet.ReflectConv(channels, channels, (1, 9), groups=channels)
            self.mix = strip_mixer_refine_unet.ReflectConv(channels, channels, 1)
            self.norm2 = nn.GroupNorm(groups, channels)
            self.ffn = nn.Sequential(
                strip_mixer_refine_unet.ReflectConv(channels, channels * 2, 1),
                nn.GELU(),
                strip_mixer_refine_unet.ReflectConv(channels * 2, channels, 1),
            )

        def forward(self, x):
            y = self.norm1(x)
            y = self.vstrip(y) + self.hstrip(y)
            x = x + self.mix(F.gelu(y))
            x = x + self.ffn(self.norm2(x))
            return x

    class DownBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            groups = 8 if out_channels % 8 == 0 else 4 if out_channels % 4 == 0 else 1
            self.block = nn.Sequential(
                strip_mixer_refine_unet.ReflectConv(in_channels, out_channels, 3),
                nn.GroupNorm(groups, out_channels),
                nn.GELU(),
                strip_mixer_refine_unet.StripMixerBlock(out_channels),
            )

        def forward(self, x):
            return self.block(x)

    class UpBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            groups = 8 if out_channels % 8 == 0 else 4 if out_channels % 4 == 0 else 1
            self.block = nn.Sequential(
                strip_mixer_refine_unet.ReflectConv(in_channels + skip_channels, out_channels, 3),
                nn.GroupNorm(groups, out_channels),
                nn.GELU(),
                strip_mixer_refine_unet.StripMixerBlock(out_channels),
            )

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return self.block(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.stem = self.ReflectConv(in_channels, channels[0], 3)

        self.encoder = nn.ModuleList()
        for i in range(depth):
            in_ch = channels[0] if i == 0 else channels[i - 1]
            self.encoder.append(self.DownBlock(in_ch, channels[i]))

        self.bottleneck = nn.Sequential(
            self.StripMixerBlock(channels[-1]),
            self.StripMixerBlock(channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self.UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3),
            nn.GELU(),
            self.ReflectConv(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        x = self.stem(x_masked)
        skips = []

        for i, block in enumerate(self.encoder):
            x = block(x)
            skips.append(x)
            if i != len(self.encoder) - 1:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = block(x, skip)

        output = self.head(x)
        if output.shape[-2:] != x_masked.shape[-2:]:
            output = F.interpolate(output, size=x_masked.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape != output.shape:
            valid = valid[:, :1].expand_as(output)

        output = output.clone()
        output[~valid] = float("nan")
        return output



