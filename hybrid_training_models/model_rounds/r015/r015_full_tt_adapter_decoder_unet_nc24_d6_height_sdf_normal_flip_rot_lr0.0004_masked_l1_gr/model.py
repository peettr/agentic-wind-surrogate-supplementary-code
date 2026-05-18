import torch
import torch.nn as nn
import torch.nn.functional as F

class tt_adapter_decoder_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
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

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        bottleneck_ch = min(channels[-1] * 2, n_c * 8)
        self.bottleneck = self._ConvBlock(channels[-1], bottleneck_ch)

        self.up_convs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        prev_ch = bottleneck_ch
        for ch in reversed(channels):
            self.up_convs.append(nn.ConvTranspose2d(prev_ch, ch, kernel_size=2, stride=2))
            self.decoder.append(self._ConvBlock(ch * 2, ch))
            prev_ch = ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for block in self.encoder:
            h = block(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, block, skip in zip(self.up_convs, self.decoder, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid = valid[:, :1].expand_as(out)

        return torch.where(valid, out, torch.full_like(out, float("nan")))