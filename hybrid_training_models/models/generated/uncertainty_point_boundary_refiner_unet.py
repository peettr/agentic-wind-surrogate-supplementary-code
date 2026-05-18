import torch
import torch.nn as nn
import torch.nn.functional as F


class uncertainty_point_boundary_refiner_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, groups=8):
            super().__init__()
            pad = kernel_size // 2
            num_groups = min(groups, out_ch)
            while out_ch % num_groups != 0:
                num_groups -= 1

            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=False),
                nn.GroupNorm(num_groups, out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = uncertainty_point_boundary_refiner_unet.ReflectConv(in_ch, out_ch)
            self.conv2 = uncertainty_point_boundary_refiner_unet.ReflectConv(out_ch, out_ch)
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, padding=0)

        def forward(self, x):
            return self.conv2(self.conv1(x)) + self.skip(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels + 1
        for ch in channels:
            self.encoders.append(self.Block(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.AvgPool2d(2)

        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            self.Block(bottleneck_ch, bottleneck_ch),
            self.Block(bottleneck_ch, bottleneck_ch),
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_ch = bottleneck_ch
        for skip_ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(dec_ch, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(self.Block(skip_ch + skip_ch, skip_ch))
            dec_ch = skip_ch

        self.boundary_head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], 1, 3, padding=0),
            nn.Sigmoid(),
        )

        self.uncertainty_head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], 1, 3, padding=0),
            nn.Sigmoid(),
        )

        self.pressure_head = nn.Sequential(
            self.ReflectConv(channels[0] + 2, channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], 1, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        valid_mask = valid.any(dim=1, keepdim=True)
        x_masked = torch.where(valid, x, torch.zeros_like(x))
        valid_float = valid_mask.to(dtype=x.dtype)

        h = torch.cat([x_masked, valid_float], dim=1)
        skips = []

        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = upconv(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = decoder(torch.cat([h, skip], dim=1))

        boundary = self.boundary_head(h)
        uncertainty = self.uncertainty_head(h)
        out = self.pressure_head(torch.cat([h, boundary, uncertainty], dim=1))

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        nan_fill = torch.full_like(out, float("nan"))
        out = torch.where(valid_mask.expand_as(out), out, nan_fill)
        return out