import torch
import torch.nn as nn
import torch.nn.functional as F


class boundary_path_message(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            groups = min(8, out_ch)
            while out_ch % groups != 0:
                groups -= 1

            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._ConvBlock(channels[-1], channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(
                nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2)
            )
            self.decoders.append(self._ConvBlock(channels[i] * 2, channels[i]))

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        skips = skips[:-1][::-1]
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, skips):
            h = upconv(h)

            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.out_conv(self.out_pad(h))

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        output_valid = valid
        if output_valid.shape[1] != output.shape[1]:
            output_valid = output_valid.all(dim=1, keepdim=True)
        if output_valid.shape[1] != output.shape[1]:
            output_valid = output_valid.expand(-1, output.shape[1], -1, -1)

        output = torch.where(output_valid, output, torch.full_like(output, float("nan")))
        return output