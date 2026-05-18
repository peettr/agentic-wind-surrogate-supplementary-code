import torch
import torch.nn as nn
import torch.nn.functional as F

class visual_context_replay_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                visual_context_replay_unet.ReflectConv(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                visual_context_replay_unet.ReflectConv(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
            self.skip = (
                visual_context_replay_unet.ReflectConv(in_ch, out_ch, 1, bias=False)
                if in_ch != out_ch else nn.Identity()
            )

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class Bottleneck(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.local = visual_context_replay_unet.ConvBlock(ch, ch)
            self.context = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, max(ch // 4, 1), 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(ch // 4, 1), ch, 1),
                nn.Sigmoid(),
            )
            self.mix = visual_context_replay_unet.ReflectConv(ch, ch, 3, bias=False)

        def forward(self, x):
            y = self.local(x)
            y = y * self.context(y)
            return self.mix(y) + x

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.Bottleneck(channels[-1])

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(self.ReflectConv(channels[i + 1], channels[i], 3, bias=False))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            self.ReflectConv(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        output = self.head(h)

        if output.shape == valid.shape:
            output = output.masked_fill(~valid, float("nan"))
        else:
            valid_out = valid.all(dim=1, keepdim=True).expand_as(output)
            output = output.masked_fill(~valid_out, float("nan"))

        return output