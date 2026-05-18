import torch
import torch.nn as nn
import torch.nn.functional as F

class physics_state_slice_mixer_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class _MixerBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.depthwise = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(channels, channels, kernel_size=3, groups=channels, padding=0, bias=False),
                nn.GroupNorm(min(8, channels), channels),
                nn.SiLU(inplace=True),
            )
            self.pointwise = nn.Sequential(
                nn.Conv2d(channels, channels * 2, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels * 2, channels, kernel_size=1),
            )
            self.scale = nn.Parameter(torch.tensor(0.1))

        def forward(self, x):
            y = self.depthwise(x)
            y = self.pointwise(y)
            return x + self.scale * y

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_proj = self._ConvBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(1, depth):
            self.down.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder.append(self._ConvBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            self._ConvBlock(channels[-1], channels[-1]),
            self._MixerBlock(channels[-1]),
            self._MixerBlock(channels[-1]),
        )

        self.up_reduce = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_reduce.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoder.append(self._ConvBlock(channels[i] * 2, channels[i]))

        self.output_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.input_proj(x_masked)
        skips.append(y)

        for down, enc in zip(self.down, self.encoder):
            y = down(y)
            y = enc(y)
            skips.append(y)

        y = self.bottleneck(y)

        for reduce, dec, skip in zip(self.up_reduce, self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = reduce(y)
            y = torch.cat([y, skip], dim=1)
            y = dec(y)

        y = self.output_head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != self.out_channels:
            valid_out = valid[:, :1].expand(-1, self.out_channels, -1, -1)
        else:
            valid_out = valid

        nan_fill = torch.full_like(y, float("nan"))
        y = torch.where(valid_out, y, nan_fill)
        return y


