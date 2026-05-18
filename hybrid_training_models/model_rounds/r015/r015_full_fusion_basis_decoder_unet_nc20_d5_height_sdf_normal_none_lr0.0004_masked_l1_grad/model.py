import torch
import torch.nn as nn
import torch.nn.functional as F


class fusion_basis_decoder_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            groups = min(8, out_channels)
            while out_channels % groups != 0:
                groups -= 1

            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
                nn.GroupNorm(groups, out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoder.append(self.ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self.ConvBlock(channels[i + 1] + channels[i], channels[i]))

        self.out_proj = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i != len(self.encoder) - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.out_proj(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            if out_valid.shape[1] == 1:
                out_valid = out_valid.expand(-1, out.shape[1], -1, -1)
            elif out.shape[1] == 1:
                out_valid = out_valid.all(dim=1, keepdim=True)
            else:
                out_valid = out_valid.all(dim=1, keepdim=True).expand(-1, out.shape[1], -1, -1)

        return torch.where(out_valid, out, torch.full_like(out, float("nan")))


