import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            _ReflectConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.net(x) + self.skip(x)

class shape_basis_residual_head_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_proj = _ReflectConv2d(in_channels, channels[0], 3, bias=False)

        self.encoder = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.encoder.append(_ConvBlock(prev, ch))
            prev = ch

        self.bottleneck = _ConvBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(
                nn.ModuleDict({
                    "reduce": _ReflectConv2d(channels[i + 1], channels[i], 3, bias=False),
                    "block": _ConvBlock(channels[i] * 2, channels[i]),
                })
            )

        self.shape_basis_head = nn.Sequential(
            _ReflectConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            _ReflectConv2d(channels[0], out_channels, 3, bias=True),
        )

        self.residual_head = nn.Sequential(
            _ReflectConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        h_in, w_in = x_masked.shape[-2:]
        h = self.input_proj(x_masked)

        skips = []
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i != self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = dec["reduce"](h)
            h = dec["block"](torch.cat([h, skip], dim=1))

        out = self.shape_basis_head(h) + self.residual_head(h)

        if out.shape[-2:] != (h_in, w_in):
            out = F.interpolate(out, size=(h_in, w_in), mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand(-1, out.shape[1], -1, -1)

        return torch.where(valid, out, torch.full_like(out, float("nan")))


