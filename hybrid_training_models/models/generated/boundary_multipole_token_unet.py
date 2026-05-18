import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class boundary_multipole_token_unet(nn.Module):
    class ReflectionConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=False),
                _gn(out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                boundary_multipole_token_unet.ReflectionConv2d(in_channels, out_channels, 3),
                boundary_multipole_token_unet.ReflectionConv2d(out_channels, out_channels, 3),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
            )

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class BoundaryTokenBlock(nn.Module):
        def __init__(self, channels, token_count=8):
            super().__init__()
            self.token_count = token_count
            self.to_tokens = nn.Linear(channels * 4, channels * token_count)
            self.mix = nn.Sequential(
                nn.Linear(channels, channels * 2),
                nn.SiLU(inplace=True),
                nn.Linear(channels * 2, channels),
            )
            self.proj = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)

        def forward(self, x):
            b, c, h, w = x.shape
            top = x[:, :, 0, :].mean(dim=-1)
            bottom = x[:, :, -1, :].mean(dim=-1)
            left = x[:, :, :, 0].mean(dim=-1)
            right = x[:, :, :, -1].mean(dim=-1)
            boundary = torch.cat([top, bottom, left, right], dim=1)
            tokens = self.to_tokens(boundary).view(b, self.token_count, c)
            tokens = self.mix(tokens).mean(dim=1).view(b, c, 1, 1)
            return x + self.proj(tokens.expand_as(x))

    class MultipoleBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.scale = nn.Sequential(
                nn.Linear(channels * 4, channels),
                nn.SiLU(inplace=True),
                nn.Linear(channels, channels),
                nn.Sigmoid(),
            )
            self.shift = nn.Sequential(
                nn.Linear(channels * 4, channels),
                nn.SiLU(inplace=True),
                nn.Linear(channels, channels),
            )

        def forward(self, x):
            b, c, h, w = x.shape
            yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
            xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
            m0 = x.mean(dim=(2, 3))
            mx = (x * xx).mean(dim=(2, 3))
            my = (x * yy).mean(dim=(2, 3))
            mxy = (x * xx * yy).mean(dim=(2, 3))
            moments = torch.cat([m0, mx, my, mxy], dim=1)
            scale = self.scale(moments).view(b, c, 1, 1)
            shift = self.shift(moments).view(b, c, 1, 1)
            return x * (1.0 + scale) + shift

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_proj = self.ConvBlock(in_channels, channels[0])
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(1, depth):
            self.downs.append(nn.AvgPool2d(kernel_size=2, stride=2))
            self.encoders.append(self.ConvBlock(channels[i - 1], channels[i]))

        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            self.ConvBlock(bottleneck_channels, bottleneck_channels),
            self.BoundaryTokenBlock(bottleneck_channels),
            self.MultipoleBlock(bottleneck_channels),
            self.ConvBlock(bottleneck_channels, bottleneck_channels),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1, padding=0, bias=False))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ReflectionConv2d(channels[0], channels[0], 3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.input_proj(x_masked)
        skips.append(y)

        for down, encoder in zip(self.downs, self.encoders):
            y = down(y)
            y = encoder(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = decoder(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid[:, :1].expand(-1, y.shape[1], -1, -1)
        else:
            valid_out = valid

        y = torch.where(valid_out, y, torch.full_like(y, float("nan")))
        return y


