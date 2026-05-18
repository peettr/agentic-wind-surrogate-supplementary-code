import torch
import torch.nn as nn
import torch.nn.functional as F

class boundary_strip_graph_mixer_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, groups=8):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=False)
            self.norm = nn.GroupNorm(min(groups, out_ch), out_ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.norm(self.conv(self.pad(x))))

    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.c1 = boundary_strip_graph_mixer_unet.RefConv(in_ch, out_ch)
            self.c2 = boundary_strip_graph_mixer_unet.RefConv(out_ch, out_ch)
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)

        def forward(self, x):
            return self.c2(self.c1(x)) + self.skip(x)

    class StripMixer(nn.Module):
        def __init__(self, ch):
            super().__init__()
            hidden = max(ch // 4, 8)
            self.row = nn.Sequential(
                nn.Conv1d(ch, hidden, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv1d(hidden, ch, 1, bias=True),
            )
            self.col = nn.Sequential(
                nn.Conv1d(ch, hidden, 1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv1d(hidden, ch, 1, bias=True),
            )
            self.gate = nn.Parameter(torch.zeros(1, ch, 1, 1))

        def forward(self, x):
            row_feat = x.mean(dim=3)
            col_feat = x.mean(dim=2)
            row_feat = self.row(row_feat).unsqueeze(3)
            col_feat = self.col(col_feat).unsqueeze(2)
            return x + self.gate * (row_feat + col_feat)

    class BoundaryMixer(nn.Module):
        def __init__(self, ch, strip=16):
            super().__init__()
            self.strip = strip
            self.mix = nn.Sequential(
                nn.Conv2d(ch, ch, 1, padding=0, bias=False),
                nn.GroupNorm(min(8, ch), ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, ch, 1, padding=0, bias=True),
            )
            self.gate = nn.Parameter(torch.zeros(1, ch, 1, 1))

        def forward(self, x):
            b, c, h, w = x.shape
            s = min(self.strip, h // 2, w // 2)
            if s < 1:
                return x

            mask = torch.zeros_like(x)
            mask[:, :, :s, :] = 1
            mask[:, :, -s:, :] = 1
            mask[:, :, :, :s] = 1
            mask[:, :, :, -s:] = 1

            denom = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            context = (x * mask).sum(dim=(2, 3), keepdim=True) / denom
            return x + self.gate * self.mix(context)

    class Bottleneck(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.local = boundary_strip_graph_mixer_unet.ResBlock(ch, ch)
            self.strip = boundary_strip_graph_mixer_unet.StripMixer(ch)
            self.boundary = boundary_strip_graph_mixer_unet.BoundaryMixer(ch)
            self.out = boundary_strip_graph_mixer_unet.ResBlock(ch, ch)

        def forward(self, x):
            x = self.local(x)
            x = self.strip(x)
            x = self.boundary(x)
            return self.out(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self.ResBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(2, 2)
        self.bottleneck = self.Bottleneck(channels[-1])

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0, bias=False))
            self.decoders.append(self.ResBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.RefConv(channels[0], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1, padding=0, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked
        for i, enc in enumerate(self.encoders):
            y = enc(y)
            skips.append(y)
            if i != len(self.encoders) - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = proj(y)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True)
        return torch.where(valid, y, torch.full_like(y, float("nan")))


