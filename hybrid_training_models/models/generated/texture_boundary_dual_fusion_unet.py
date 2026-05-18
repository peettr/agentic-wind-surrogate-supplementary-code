import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv(in_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
            _ReflectConv(out_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.net(x) + self.skip(x)


class _AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super().__init__()
        self.gate_proj = nn.Conv2d(gate_channels, inter_channels, 1, padding=0, bias=False)
        self.skip_proj = nn.Conv2d(skip_channels, inter_channels, 1, padding=0, bias=False)
        self.psi = nn.Conv2d(inter_channels, 1, 1, padding=0, bias=True)

    def forward(self, gate, skip):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attn = torch.sigmoid(self.psi(F.silu(self.gate_proj(gate) + self.skip_proj(skip))))
        return skip * attn


class texture_boundary_dual_fusion_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(2, int(depth))
        self.in_channels = in_channels
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        stem_in = in_channels * 3
        self.input_stem = _ConvBlock(stem_in, channels[0])

        self.encoders = nn.ModuleList()
        for i in range(1, depth):
            self.encoders.append(_ConvBlock(channels[i - 1], channels[i]))

        self.down = nn.AvgPool2d(2, 2)
        self.bottleneck = _ConvBlock(channels[-1], channels[-1])

        self.texture_head = nn.Sequential(
            _ReflectConv(channels[-1], channels[-1], 3, bias=False),
            _gn(channels[-1]),
            nn.SiLU(inplace=True),
        )
        self.boundary_head = nn.Sequential(
            _ReflectConv(channels[-1], channels[-1], 3, bias=False),
            _gn(channels[-1]),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Conv2d(channels[-1] * 2, channels[-1], 1, padding=0, bias=False)

        self.attn = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            inter = max(channels[i] // 2, 1)
            self.attn.append(_AttentionGate(channels[i + 1], channels[i], inter))
            self.decoders.append(_ConvBlock(channels[i + 1] + channels[i], channels[i]))

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=0, bias=True)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        valid_f = valid.to(x.dtype)
        denom = valid_f.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (x_masked * valid_f).sum(dim=(2, 3), keepdim=True) / denom
        centered = (x_masked - mean) * valid_f
        gx = F.pad(centered[:, :, :, 1:] - centered[:, :, :, :-1], (0, 1, 0, 0), mode="reflect")
        gy = F.pad(centered[:, :, 1:, :] - centered[:, :, :-1, :], (0, 0, 0, 1), mode="reflect")
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)

        h = self.input_stem(torch.cat([x_masked, edge, valid_f], dim=1))
        skips = [h]

        for enc in self.encoders:
            h = self.down(h)
            h = enc(h)
            skips.append(h)

        h = self.bottleneck(h)
        texture = self.texture_head(h)
        boundary = self.boundary_head(h)
        h = self.fusion(torch.cat([texture, boundary], dim=1))

        for attn, dec, skip in zip(self.attn, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = attn(h, skip)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.out_conv(self.out_pad(h))
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            if out_valid.shape[1] == 1:
                out_valid = out_valid.expand(-1, out.shape[1], -1, -1)
            else:
                out_valid = out_valid.any(dim=1, keepdim=True).expand(-1, out.shape[1], -1, -1)
        return torch.where(out_valid, out, torch.full_like(out, float("nan")))


