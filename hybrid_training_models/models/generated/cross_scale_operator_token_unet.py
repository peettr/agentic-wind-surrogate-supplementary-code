import torch
import torch.nn as nn
import torch.nn.functional as F

class cross_scale_operator_token_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                cross_scale_operator_token_unet.ReflectConv(in_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                cross_scale_operator_token_unet.ReflectConv(out_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class BottleneckOperator(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.local = cross_scale_operator_token_unet.ConvBlock(channels, channels)
            self.q = nn.Conv2d(channels, channels, kernel_size=1)
            self.k = nn.Conv2d(channels, channels, kernel_size=1)
            self.v = nn.Conv2d(channels, channels, kernel_size=1)
            self.proj = nn.Conv2d(channels, channels, kernel_size=1)
            self.norm = nn.GroupNorm(min(8, channels), channels)

        def forward(self, x):
            residual = x
            x = self.local(x)

            pooled = F.adaptive_avg_pool2d(x, (16, 16))
            q = self.q(pooled)
            k = self.k(pooled)
            v = self.v(pooled)

            b, c, h, w = q.shape
            q = q.flatten(2).transpose(1, 2)
            k = k.flatten(2)
            v = v.flatten(2).transpose(1, 2)

            attn = torch.softmax(torch.bmm(q, k) * (c ** -0.5), dim=-1)
            y = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
            y = self.proj(y)

            return self.norm(x + y + residual)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
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

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.BottleneckOperator(channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x

        for i, enc in enumerate(self.encoders):
            y = enc(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.pool(y)

        y = self.bottleneck(y)

        skips = skips[:-1][::-1]
        for up, dec, skip in zip(self.upconvs, self.decoders, skips):
            y = up(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = dec(y)

        y = self.out_conv(self.out_pad(y))

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid = valid.expand(-1, y.shape[1], -1, -1)
        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y


