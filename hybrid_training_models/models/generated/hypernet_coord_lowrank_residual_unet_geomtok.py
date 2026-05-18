import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for num_groups in range(min(8, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class hypernet_coord_lowrank_residual_unet_geomtok(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, bias=True):
            super().__init__()
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.norm1 = _gn(out_ch)
            self.norm2 = _gn(out_ch)
            self.conv_in = hypernet_coord_lowrank_residual_unet_geomtok.ReflectConv(in_ch, out_ch, 3)
            self.conv1 = hypernet_coord_lowrank_residual_unet_geomtok.ReflectConv(out_ch, out_ch, 3)
            self.conv2 = hypernet_coord_lowrank_residual_unet_geomtok.ReflectConv(out_ch, out_ch, 3)
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):
            h = self.conv_in(x)
            h = self.conv1(F.silu(self.norm1(h)))
            h = self.conv2(F.silu(self.norm2(h)))
            return h + self.skip(x)

    class CoordLowRankGate(nn.Module):
        def __init__(self, channels, rank=8):
            super().__init__()
            rank = min(rank, channels)
            self.to_rank = nn.Conv2d(4, rank, 1)
            self.mix = nn.Conv2d(rank, channels, 1)
            self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x):
            b, c, h, w = x.shape
            yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
            xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
            coords = torch.cat((xx, yy, xx * yy, xx.square() + yy.square()), dim=1)
            gate = torch.tanh(self.mix(F.silu(self.to_rank(coords))))
            return x * (1.0 + self.scale * gate)

    class GeomTokenBlock(nn.Module):
        def __init__(self, channels, tokens=8):
            super().__init__()
            self.tokens = nn.Parameter(torch.randn(1, tokens, channels) * 0.02)
            self.q = nn.Linear(channels, channels)
            self.k = nn.Linear(channels, channels)
            self.v = nn.Linear(channels, channels)
            self.proj = nn.Linear(channels, channels)
            self.norm = nn.LayerNorm(channels)
            self.scale = channels ** -0.5

        def forward(self, x):
            b, c, h, w = x.shape
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            tokens = self.tokens.expand(b, -1, -1) + pooled.unsqueeze(1)
            q = self.q(tokens)
            k = self.k(tokens)
            v = self.v(tokens)
            attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
            tokens = self.norm(tokens + self.proj(torch.matmul(attn, v)))
            geom = tokens.mean(dim=1).view(b, c, 1, 1)
            return x + geom

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self.ReflectConv(in_channels, channels[0], 3)

        self.encoders = nn.ModuleList()
        self.coord_gates = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.encoders.append(self.ResBlock(prev, ch))
            self.coord_gates.append(self.CoordLowRankGate(ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            self.ResBlock(channels[-1], channels[-1]),
            self.GeomTokenBlock(channels[-1]),
            self.ResBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        self.up_proj = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_proj.append(nn.Conv2d(channels[i + 1], channels[i], 1))
            self.decoders.append(self.ResBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ResBlock(channels[0], channels[0]),
            self.ReflectConv(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        h = self.stem(x_masked)
        skips = []

        for i, (enc, gate) in enumerate(zip(self.encoders, self.coord_gates)):
            h = gate(enc(h))
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = self.up_proj[i](h)
            h = dec(torch.cat((h, skip), dim=1))

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid.expand(-1, out.shape[1], -1, -1)
        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out