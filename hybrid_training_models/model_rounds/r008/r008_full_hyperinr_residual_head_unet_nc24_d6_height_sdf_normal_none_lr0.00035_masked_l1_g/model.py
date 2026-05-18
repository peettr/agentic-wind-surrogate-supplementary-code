import torch
import torch.nn as nn
import torch.nn.functional as F

class hyperinr_residual_head_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        self.up_proj = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_proj.append(nn.Conv2d(channels[i + 1], channels[i], 1))
            self.decoder.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        input_size = x_masked.shape[-2:]

        skips = []
        h = x_masked
        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for proj, block, skip in zip(self.up_proj, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = proj(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        out = out + self.residual(x_masked)

        if valid.shape[1] != out.shape[1]:
            valid = valid[:, :1].expand(-1, out.shape[1], -1, -1)
        out = torch.where(valid, out, torch.full_like(out, float("nan")))
        return out