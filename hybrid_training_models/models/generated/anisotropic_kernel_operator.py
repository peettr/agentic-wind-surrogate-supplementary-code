import torch
import torch.nn as nn
import torch.nn.functional as F


class anisotropic_kernel_operator(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                bias=bias,
            )

        def forward(self, x):
            return self.conv(self.pad(x))

    class Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            groups = min(8, out_channels)
            while out_channels % groups != 0:
                groups -= 1

            self.net = nn.Sequential(
                anisotropic_kernel_operator.RefConv(in_channels, out_channels, 3),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
                anisotropic_kernel_operator.RefConv(out_channels, out_channels, 3),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
            )

            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else anisotropic_kernel_operator.RefConv(in_channels, out_channels, 1)
            )

        def forward(self, x):
            return self.net(x) + self.skip(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        prev = in_channels
        for ch in channels:
            self.encoders.append(self.Block(prev, ch))
            prev = ch

        for i in range(depth - 1):
            self.downs.append(self.RefConv(channels[i], channels[i], 3, stride=2))

        self.bottleneck = nn.Sequential(
            self.Block(channels[-1], channels[-1]),
            self.Block(channels[-1], channels[-1]),
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            self.up_convs.append(self.RefConv(channels[i + 1], channels[i], 3))
            self.decoders.append(self.Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.RefConv(channels[0], channels[0], 3),
            nn.SiLU(inplace=True),
            self.RefConv(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x).any(dim=1, keepdim=True)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        h = self.bottleneck(h)

        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_conv(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        nan_fill = torch.full_like(output, float("nan"))
        output = torch.where(valid.expand_as(output), output, nan_fill)
        return output


