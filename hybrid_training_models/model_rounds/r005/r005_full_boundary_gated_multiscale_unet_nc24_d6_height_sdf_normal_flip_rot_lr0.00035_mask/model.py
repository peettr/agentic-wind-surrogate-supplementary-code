import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for num_groups in range(min(8, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class boundary_gated_multiscale_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, bias=False):
            super().__init__()
            pad = dilation * (kernel_size // 2)
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=0,
                dilation=dilation,
                bias=bias,
            )

        def forward(self, x):
            return self.conv(self.pad(x))

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.c1 = boundary_gated_multiscale_unet.ReflectConv(in_ch, out_ch)
            self.n1 = _gn(out_ch)
            self.c2 = boundary_gated_multiscale_unet.ReflectConv(out_ch, out_ch)
            self.n2 = _gn(out_ch)
            self.d2 = boundary_gated_multiscale_unet.ReflectConv(out_ch, out_ch, dilation=2)
            self.proj = (
                nn.Identity()
                if in_ch == out_ch
                else nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
            )

        def forward(self, x):
            r = self.proj(x)
            y = F.silu(self.n1(self.c1(x)))
            y = self.n2(self.c2(y) + self.d2(y))
            return F.silu(y + r)

    class Gate(nn.Module):
        def __init__(self, skip_ch, dec_ch, edge_ch):
            super().__init__()
            self.skip_proj = nn.Conv2d(skip_ch, skip_ch, kernel_size=1, padding=0)
            self.dec_proj = nn.Conv2d(dec_ch, skip_ch, kernel_size=1, padding=0)
            self.edge_proj = nn.Conv2d(edge_ch, skip_ch, kernel_size=1, padding=0)
            self.out = nn.Conv2d(skip_ch, skip_ch, kernel_size=1, padding=0)

        def forward(self, skip, dec, edge):
            if dec.shape[-2:] != skip.shape[-2:]:
                dec = F.interpolate(dec, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if edge.shape[-2:] != skip.shape[-2:]:
                edge = F.interpolate(edge, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            g = self.skip_proj(skip) + self.dec_proj(dec) + self.edge_proj(edge)
            g = torch.sigmoid(self.out(F.silu(g)))
            return skip * g

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self.Block(prev, ch))
            prev = ch

        self.bottleneck = self.Block(channels[-1], channels[-1])

        self.gates = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.gates.append(self.Gate(channels[i], channels[i + 1], in_channels))
            self.up_blocks.append(self.Block(channels[i + 1] + channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0]),
            _gn(channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def _edge_map(self, x):
        gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
        gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
        edge = gx + gy
        return edge / (edge.amax(dim=(-2, -1), keepdim=True) + 1e-6)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        edge = self._edge_map(x_masked)

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for i, block in enumerate(self.up_blocks):
            skip = skips[-(i + 2)]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            gated_skip = self.gates[i](skip, h, edge)
            h = block(torch.cat([h, gated_skip], dim=1))

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            out_valid = out_valid.all(dim=1, keepdim=True).expand_as(out)

        return torch.where(out_valid, out, torch.full_like(out, float("nan")))


