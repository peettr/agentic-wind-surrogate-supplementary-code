import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, groups=1):
        super().__init__()
        p = k // 2
        self.pad = nn.ReflectionPad2d(p)
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=0, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.pad(x))

class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        g1 = min(8, out_ch)
        while out_ch % g1 != 0:
            g1 -= 1
        self.net = nn.Sequential(
            _ReflectConv(in_ch, out_ch, 3),
            nn.GroupNorm(g1, out_ch),
            nn.SiLU(inplace=True),
            _ReflectConv(out_ch, out_ch, 3),
            nn.GroupNorm(g1, out_ch),
            nn.SiLU(inplace=True),
        )
        self.short = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.net(x) + self.short(x)

class _SkipMixer(nn.Module):
    def __init__(self, ch):
        super().__init__()
        g = min(8, ch)
        while ch % g != 0:
            g -= 1
        self.local = nn.Sequential(
            _ReflectConv(ch, ch, 3, groups=ch),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.GroupNorm(g, ch),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(1, ch // 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, ch // 4), ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.local(x)
        return x + y * self.gate(y)

class omniscan_skip_mixer_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        chans = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.encoders = nn.ModuleList()
        self.mixers = nn.ModuleList()

        prev = in_channels
        for ch in chans:
            self.encoders.append(_ConvBlock(prev, ch))
            self.mixers.append(_SkipMixer(ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            _ConvBlock(chans[-1], chans[-1]),
            _SkipMixer(chans[-1]),
            _ConvBlock(chans[-1], chans[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(chans):
            self.up_projs.append(nn.Conv2d(prev, ch, 1, bias=False))
            self.decoders.append(_ConvBlock(ch * 2, ch))
            prev = ch

        self.out_head = nn.Sequential(
            _ReflectConv(chans[0], chans[0], 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(chans[0], out_channels, 1)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for enc, mix in zip(self.encoders, self.mixers):
            h = mix(enc(h))
            skips.append(h)
            h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = proj(h)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.out_head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != out.shape[1]:
            valid_out = valid_out[:, :1].expand(-1, out.shape[1], -1, -1)
        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


