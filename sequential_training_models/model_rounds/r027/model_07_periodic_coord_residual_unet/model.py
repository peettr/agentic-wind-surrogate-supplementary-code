import torch
import torch.nn as nn
import torch.nn.functional as F

class periodic_coord_residual_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels + 4
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1, bias=False))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def _coords(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xx = torch.linspace(0.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        yy = yy.expand(b, 1, h, w)
        xx = xx.expand(b, 1, h, w)
        return torch.cat((
            torch.sin(2.0 * torch.pi * xx),
            torch.cos(2.0 * torch.pi * xx),
            torch.sin(2.0 * torch.pi * yy),
            torch.cos(2.0 * torch.pi * yy),
        ), dim=1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        valid_out = valid.all(dim=1, keepdim=True)
        if self.out_channels != 1:
            valid_out = valid_out.expand(-1, self.out_channels, -1, -1)

        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        h = torch.cat([x_masked, self._coords(x_masked)], dim=1)

        skips = []
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels == x_masked.shape[1]:
            out = out + x_masked

        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


