import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class multi_scale_fourier_basis_head_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class FourierHead(nn.Module):
        def __init__(self, in_ch, out_ch, width=16):
            super().__init__()
            self.width = width
            self.proj = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch + 4, in_ch, 3, bias=False),
                _gn(in_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1),
            )

        def forward(self, x):
            b, _, h, w = x.shape
            yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
            xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)

            scale = float(self.width)
            basis = torch.cat((
                torch.sin(scale * xx),
                torch.cos(scale * xx),
                torch.sin(scale * yy),
                torch.cos(scale * yy),
            ), dim=1)

            return self.proj(torch.cat((x, basis), dim=1))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.AvgPool2d(2)
        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.up = nn.ModuleList()
        self.decoder = nn.ModuleList()
        dec_in = channels[-1]
        for skip_ch in reversed(channels):
            self.up.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            self.decoder.append(self.ConvBlock(dec_in + skip_ch, skip_ch))
            dec_in = skip_ch

        self.head = self.FourierHead(channels[0], out_channels)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked
        for block in self.encoder:
            y = block(y)
            skips.append(y)
            y = self.pool(y)

        y = self.bottleneck(y)

        for up, block, skip in zip(self.up, self.decoder, reversed(skips)):
            y = up(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat((y, skip), dim=1))

        y = self.head(y)
        out_valid = valid if valid.shape[1] == y.shape[1] else valid[:, :1].expand_as(y)
        y = torch.where(out_valid, y, torch.full_like(y, float("nan")))
        return y