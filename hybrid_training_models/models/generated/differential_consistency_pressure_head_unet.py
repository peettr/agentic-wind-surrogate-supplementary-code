import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class differential_consistency_pressure_head_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._ConvBlock(channels[-1], channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, ch, kernel_size=2, stride=2))
            self.decoders.append(self._ConvBlock(ch + ch, ch))
            prev_ch = ch

        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = upconv(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.final(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid[:, :1].expand_as(output)

        output = output.clone()
        output[~valid] = float("nan")
        return output


