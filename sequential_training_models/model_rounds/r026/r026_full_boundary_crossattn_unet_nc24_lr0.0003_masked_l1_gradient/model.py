import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias),
        )

    def forward(self, x):
        return self.net(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            ReflectConv(in_channels, out_channels, 3),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            ReflectConv(out_channels, out_channels, 3),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class BoundaryCrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 4)
        self.q = nn.Conv2d(channels, hidden, 1, bias=False)
        self.k = nn.Conv2d(channels, hidden, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.out = nn.Conv2d(channels, channels, 1, bias=True)
        self.boundary = nn.Sequential(
            ReflectConv(channels, hidden, 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, skip, decoder):
        if decoder.shape[-2:] != skip.shape[-2:]:
            decoder = F.interpolate(decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        q = self.q(decoder)
        k = self.k(skip)
        v = self.v(skip)

        attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / (q.shape[1] ** 0.5))
        edge_gate = self.boundary(skip)
        return skip + self.out(v * attn * edge_gate)

class boundary_crossattn_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.up_projs = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.decoders = nn.ModuleList()

        dec_in = channels[-1]
        for skip_ch in reversed(channels):
            self.up_projs.append(nn.Conv2d(dec_in, skip_ch, 1, bias=False))
            self.attn.append(BoundaryCrossAttention(skip_ch))
            self.decoders.append(ConvBlock(skip_ch * 2, skip_ch))
            dec_in = skip_ch

        self.head = nn.Sequential(
            ReflectConv(channels[0], channels[0], 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up_proj, attn, dec, skip in zip(self.up_projs, self.attn, self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            skip = attn(skip, h)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            out_valid = out_valid.expand(-1, out.shape[1], -1, -1)
        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out