import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class coord_basis_pressure_head_unet(nn.Module):
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
        self.coord_channels = 6

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        enc_in = in_channels + self.coord_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(enc_in, ch))
            enc_in = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._ConvBlock(channels[-1], channels[-1])

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.up_convs.append(nn.Conv2d(dec_in, skip_ch, kernel_size=1))
            self.decoders.append(self._ConvBlock(skip_ch + skip_ch, skip_ch))
            dec_in = skip_ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0] + self.coord_channels, channels[0], kernel_size=3, padding=0),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def _coord_basis(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        yy = yy.expand(b, 1, h, w)
        xx = xx.expand(b, 1, h, w)
        rr = torch.sqrt(torch.clamp(xx * xx + yy * yy, min=0.0))
        return torch.cat(
            [
                xx,
                yy,
                rr,
                torch.sin(3.141592653589793 * xx),
                torch.sin(3.141592653589793 * yy),
                torch.cos(3.141592653589793 * rr),
            ],
            dim=1,
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        coords = self._coord_basis(x_masked)
        h = torch.cat([x_masked, coords], dim=1)

        skips = []
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1))

        h = F.interpolate(h, size=x.shape[-2:], mode="bilinear", align_corners=False)
        out = self.head(torch.cat([h, coords], dim=1))

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = out.clone()
        out[~valid.expand_as(out)] = float("nan")
        return out