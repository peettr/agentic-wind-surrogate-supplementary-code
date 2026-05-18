import torch
import torch.nn as nn
import torch.nn.functional as F


class boundary_alltoall_surface_mixer_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                boundary_alltoall_surface_mixer_unet.ReflectConv(in_channels, out_channels, 3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                boundary_alltoall_surface_mixer_unet.ReflectConv(out_channels, out_channels, 3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
            )

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class SurfaceMixer(nn.Module):
        def __init__(self, channels, reduction=8):
            super().__init__()
            hidden = max(channels // reduction, 4)
            self.boundary_proj = nn.Sequential(
                nn.Linear(channels * 4, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, channels),
                nn.Sigmoid(),
            )
            self.context_proj = nn.Sequential(
                nn.Conv2d(channels, hidden, 1, padding=0, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, 1, padding=0, bias=True),
                nn.Sigmoid(),
            )
            self.mix = nn.Conv2d(channels, channels, 1, padding=0, bias=True)

        def forward(self, x):
            top = x[:, :, 0, :].mean(dim=-1)
            bottom = x[:, :, -1, :].mean(dim=-1)
            left = x[:, :, :, 0].mean(dim=-1)
            right = x[:, :, :, -1].mean(dim=-1)
            boundary = torch.cat([top, bottom, left, right], dim=1)
            boundary_gate = self.boundary_proj(boundary).view(x.shape[0], x.shape[1], 1, 1)
            context_gate = self.context_proj(F.adaptive_avg_pool2d(x, 1))
            return x + self.mix(x * boundary_gate * context_gate)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.ConvBlock(in_channels, channels[0])
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(1, depth):
            self.downs.append(nn.AvgPool2d(kernel_size=2, stride=2))
            self.encoders.append(self.ConvBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.SurfaceMixer(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0, bias=False))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, padding=0, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for down, encoder in zip(self.downs, self.encoders):
            h = down(h)
            h = encoder(h)
            skips.append(h)

        h = self.bottleneck(h)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] == output.shape[1]:
            output_valid = valid
        else:
            output_valid = valid.all(dim=1, keepdim=True)

        output = output.clone()
        output[~output_valid.expand_as(output)] = float("nan")
        return output


