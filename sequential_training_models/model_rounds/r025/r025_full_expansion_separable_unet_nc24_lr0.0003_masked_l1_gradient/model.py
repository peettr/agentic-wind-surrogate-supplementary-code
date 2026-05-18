import torch
import torch.nn as nn
import torch.nn.functional as F

class expansion_separable_unet(nn.Module):
    class SepConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, expansion=2):
            super().__init__()
            mid_ch = min(out_ch * expansion, out_ch * 4)
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, in_ch, kernel_size=3, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, mid_ch), mid_ch),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(mid_ch, mid_ch, kernel_size=3, groups=mid_ch, bias=False),
                nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.depth = depth
        self.out_channels = out_channels

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(self.SepConvBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            self.SepConvBlock(channels[-1], channels[-1]),
            self.SepConvBlock(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        dec_ch = channels[-1]
        for skip_ch in reversed(channels):
            self.up_projs.append(nn.Conv2d(dec_ch, skip_ch, kernel_size=1, bias=False))
            self.decoder.append(self.SepConvBlock(skip_ch * 2, skip_ch))
            dec_ch = skip_ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for block in self.encoder:
            h = block(h)
            skips.append(h)
            h = self.down(h)

        h = self.bottleneck(h)

        for up_proj, block, skip in zip(self.up_projs, self.decoder, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != self.out_channels:
            valid_out = valid.any(dim=1, keepdim=True).expand(-1, self.out_channels, -1, -1)
        else:
            valid_out = valid

        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out