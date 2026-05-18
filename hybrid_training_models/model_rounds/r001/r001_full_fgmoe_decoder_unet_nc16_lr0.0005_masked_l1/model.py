import torch
import torch.nn as nn
import torch.nn.functional as F

class fgmoe_decoder_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                fgmoe_decoder_unet.ReflectConv(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                fgmoe_decoder_unet.ReflectConv(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class DecoderMoEBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch, n_experts=4):
            super().__init__()
            self.pre = fgmoe_decoder_unet.ConvBlock(in_ch + skip_ch, out_ch)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    fgmoe_decoder_unet.ReflectConv(out_ch, out_ch, 3, bias=False),
                    nn.GroupNorm(min(8, out_ch), out_ch),
                    nn.SiLU(inplace=True),
                    fgmoe_decoder_unet.ReflectConv(out_ch, out_ch, 3, bias=False),
                )
                for _ in range(n_experts)
            ])
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, max(out_ch // 4, 1), 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(out_ch // 4, 1), n_experts, 1),
            )
            self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x, skip):
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.pre(x)
            weights = torch.softmax(self.gate(x), dim=1)
            y = 0
            for i, expert in enumerate(self.experts):
                y = y + expert(x) * weights[:, i:i + 1]
            return self.act(self.norm(x + y))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=5):
        super().__init__()
        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self.DecoderMoEBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = decoder(h, skip)

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = out.masked_fill(~valid, float("nan"))
        return out