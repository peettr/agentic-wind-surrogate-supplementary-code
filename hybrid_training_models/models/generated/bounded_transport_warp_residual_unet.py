import torch
import torch.nn as nn
import torch.nn.functional as F

class bounded_transport_warp_residual_unet(nn.Module):
    class _ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class _ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            groups = 1
            for g in (8, 4, 2, 1):
                if out_channels % g == 0:
                    groups = g
                    break

            self.conv1 = bounded_transport_warp_residual_unet._ReflectConv(in_channels, out_channels)
            self.norm1 = nn.GroupNorm(groups, out_channels)
            self.conv2 = bounded_transport_warp_residual_unet._ReflectConv(out_channels, out_channels)
            self.norm2 = nn.GroupNorm(groups, out_channels)
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        def forward(self, x):
            y = F.gelu(self.norm1(self.conv1(x)))
            y = self.norm2(self.conv2(y))
            return F.gelu(y + self.skip(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self._ResBlock(prev, ch))
            prev = ch

        self.bottleneck = self._ResBlock(channels[-1], channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i + 1], channels[i], 2, stride=2))
            self.decoders.append(self._ResBlock(channels[i] + channels[i], channels[i]))

        self.pressure_head = nn.Sequential(
            self._ReflectConv(channels[0], channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 1)
        )

        self.flow_head = nn.Sequential(
            self._ReflectConv(channels[0], channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], 2, 1)
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.flow_scale = nn.Parameter(torch.tensor(0.05))

    def _warp(self, x, flow):
        b, c, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
        base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, h, w, 2)
        flow = flow.permute(0, 2, 3, 1)
        grid = base_grid + flow
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i < self.depth - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips[:-1])):
            y = upconv(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = decoder(torch.cat([y, skip], dim=1))

        pressure_delta = self.pressure_head(y)
        flow = torch.tanh(self.flow_head(y)) * torch.abs(self.flow_scale)
        transported = self._warp(x_masked, flow)
        out = transported + pressure_delta * torch.abs(self.residual_scale)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = torch.where(valid, out, torch.full_like(out, float("nan")))
        return out