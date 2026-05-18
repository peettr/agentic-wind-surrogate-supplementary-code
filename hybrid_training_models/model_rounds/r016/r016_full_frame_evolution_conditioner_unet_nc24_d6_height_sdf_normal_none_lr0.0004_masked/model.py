import torch
import torch.nn as nn
import torch.nn.functional as F

class frame_evolution_conditioner_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(self._ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._ConvBlock(channels[-1], channels[-1])

        self.up_reduce = nn.ModuleList()
        self.decoder = nn.ModuleList()
        dec_in = channels[-1]
        for skip_ch in reversed(channels):
            self.up_reduce.append(nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(dec_in, skip_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, skip_ch), skip_ch),
                nn.SiLU(inplace=True),
            ))
            self.decoder.append(self._ConvBlock(skip_ch + skip_ch, skip_ch))
            dec_in = skip_ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for block in self.encoder:
            h = block(h)
            skips.append(h)
            h = self.down(h)

        h = self.bottleneck(h)

        for reduce, block, skip in zip(self.up_reduce, self.decoder, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = reduce(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != out.shape[1]:
            valid_out = valid_out[:, :1].expand(-1, out.shape[1], -1, -1)

        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


