import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class distilled_compact_inr_pressure_head_unet(nn.Module):
    class _ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class _Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                distilled_compact_inr_pressure_head_unet._ReflectConv(in_ch, out_ch, 3),
                distilled_compact_inr_pressure_head_unet._ReflectConv(out_ch, out_ch, 3),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoder = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoder.append(self._Block(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            self._Block(channels[-1], channels[-1]),
            self._Block(channels[-1], channels[-1]),
        )

        self.up_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_proj.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoder.append(self._Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self._ReflectConv(channels[0], channels[0], 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        restore_valid = valid if self.out_channels == self.in_channels else valid.all(dim=1, keepdim=True)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for up, block, skip in zip(self.up_proj, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = block(torch.cat([h, skip], dim=1))

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if restore_valid.shape[1] != out.shape[1]:
            restore_valid = restore_valid.expand(-1, out.shape[1], -1, -1)
        return torch.where(restore_valid, out, torch.full_like(out, float("nan")))