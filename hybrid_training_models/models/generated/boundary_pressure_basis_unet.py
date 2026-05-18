import torch
import torch.nn as nn
import torch.nn.functional as F


class boundary_pressure_basis_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_c = in_channels
        for ch in channels:
            self.encoders.append(self._conv_block(prev_c, ch))
            prev_c = ch

        self.bottleneck = self._conv_block(channels[-1], channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(
                nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2)
            )
            self.decoders.append(self._conv_block(channels[i] * 2, channels[i]))

        self.final = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def _groups(self, channels):
        for g in (8, 6, 4, 3, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(self._groups(out_c), out_c),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(self._groups(out_c), out_c),
            nn.GELU(),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i != self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips[:-1])):
            h = upconv(h)
            if h.shape[-2:] != skip.shape[-2:]:
                dh = skip.shape[-2] - h.shape[-2]
                dw = skip.shape[-1] - h.shape[-1]
                h = F.pad(h, (0, dw, 0, dh), mode="reflect")
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.final(h)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True)
            if output.shape[1] != 1:
                valid = valid.expand(-1, output.shape[1], -1, -1)

        output = output.clone()
        output[~valid] = float("nan")
        return output