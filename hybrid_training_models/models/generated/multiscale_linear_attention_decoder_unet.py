import torch
import torch.nn as nn
import torch.nn.functional as F

class multiscale_linear_attention_decoder_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, groups=1, bias=True):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, groups=groups, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                multiscale_linear_attention_decoder_unet.ReflectConv(in_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                multiscale_linear_attention_decoder_unet.ReflectConv(out_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class LinearAttention(nn.Module):
        def __init__(self, channels):
            super().__init__()
            heads = max(1, min(8, channels // 16))
            while channels % heads != 0:
                heads -= 1
            self.heads = heads
            self.dim_head = channels // heads
            self.to_q = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
            self.to_k = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
            self.to_v = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
            self.proj = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
            self.norm = nn.GroupNorm(min(8, channels), channels)

        def forward(self, x):
            b, c, h, w = x.shape
            n = h * w
            q = self.to_q(x).view(b, self.heads, self.dim_head, n)
            k = self.to_k(x).view(b, self.heads, self.dim_head, n)
            v = self.to_v(x).view(b, self.heads, self.dim_head, n)

            q = F.softmax(q, dim=2)
            k = F.softmax(k, dim=3)

            context = torch.matmul(k, v.transpose(-1, -2))
            out = torch.matmul(context.transpose(-1, -2), q).view(b, c, h, w)
            return x + self.norm(self.proj(out))

    class DecoderBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.fuse = multiscale_linear_attention_decoder_unet.ConvBlock(in_ch + skip_ch, out_ch)
            self.attn_full = multiscale_linear_attention_decoder_unet.LinearAttention(out_ch)
            self.attn_half = multiscale_linear_attention_decoder_unet.LinearAttention(out_ch)
            self.mix = nn.Conv2d(out_ch * 2, out_ch, 1, padding=0, bias=False)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.fuse(torch.cat([x, skip], dim=1))
            full = self.attn_full(x)
            half = F.avg_pool2d(x, kernel_size=2, stride=2) if min(x.shape[-2:]) >= 2 else x
            half = self.attn_half(half)
            half = F.interpolate(half, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return self.mix(torch.cat([full, half], dim=1))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.LinearAttention(channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self.DecoderBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked
        for i, enc in enumerate(self.encoders):
            y = enc(y)
            skips.append(y)
            if i != len(self.encoders) - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            y = dec(y, skip)

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid = valid.expand(-1, y.shape[1], -1, -1)
        y = y.clone()
        y[~valid] = float("nan")
        return y


