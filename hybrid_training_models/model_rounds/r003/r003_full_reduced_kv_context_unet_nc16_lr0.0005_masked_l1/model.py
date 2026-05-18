import torch
import torch.nn as nn
import torch.nn.functional as F

class reduced_kv_context_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias),
            )

        def forward(self, x):
            return self.net(x)

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                reduced_kv_context_unet.ReflectConv(in_channels, out_channels),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                reduced_kv_context_unet.ReflectConv(out_channels, out_channels),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class ReducedKVContext(nn.Module):
        def __init__(self, channels, heads=4, reduction=4):
            super().__init__()
            self.channels = channels
            self.heads = min(heads, channels)
            while channels % self.heads != 0:
                self.heads -= 1
            self.head_dim = channels // self.heads
            self.scale = self.head_dim ** -0.5
            self.reduction = reduction

            self.q = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
            self.k = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
            self.v = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
            self.proj = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
            self.norm = nn.GroupNorm(min(8, channels), channels)

        def forward(self, x):
            b, c, h, w = x.shape
            q = self.q(x).reshape(b, self.heads, self.head_dim, h * w).transpose(-2, -1)

            kv = F.avg_pool2d(x, kernel_size=self.reduction, stride=self.reduction, ceil_mode=True)
            kh, kw = kv.shape[-2:]
            k = self.k(kv).reshape(b, self.heads, self.head_dim, kh * kw)
            v = self.v(kv).reshape(b, self.heads, self.head_dim, kh * kw).transpose(-2, -1)

            attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)
            out = torch.matmul(attn, v).transpose(-2, -1).reshape(b, c, h, w)
            return x + self.norm(self.proj(out))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.ReducedKVContext(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self.ConvBlock(channels[i + 1] + channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0]),
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
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output