import torch
import torch.nn as nn
import torch.nn.functional as F

class swin_unet_lite(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class WindowAttentionBlock(nn.Module):
        def __init__(self, channels, window_size=8, num_heads=4):
            super().__init__()
            self.channels = channels
            self.window_size = window_size
            self.num_heads = min(num_heads, channels)
            while channels % self.num_heads != 0:
                self.num_heads -= 1

            self.norm1 = nn.GroupNorm(1, channels)
            self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
            self.proj = nn.Conv2d(channels, channels, 1, bias=False)

            self.norm2 = nn.GroupNorm(1, channels)
            self.mlp = nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1),
                nn.GELU(),
                nn.Conv2d(channels * 2, channels, 1),
            )

        def forward(self, x):
            b, c, h, w = x.shape
            ws = self.window_size

            pad_h = (ws - h % ws) % ws
            pad_w = (ws - w % ws) % ws
            y = self.norm1(x)
            if pad_h or pad_w:
                y = F.pad(y, (0, pad_w, 0, pad_h), mode="reflect")

            hp, wp = y.shape[-2:]
            qkv = self.qkv(y)
            qkv = qkv.view(b, 3, self.num_heads, c // self.num_heads, hp // ws, ws, wp // ws, ws)
            qkv = qkv.permute(1, 0, 4, 6, 2, 5, 7, 3).contiguous()
            qkv = qkv.view(3, b * (hp // ws) * (wp // ws), self.num_heads, ws * ws, c // self.num_heads)

            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = torch.matmul(q, k.transpose(-2, -1)) * ((c // self.num_heads) ** -0.5)
            attn = attn.softmax(dim=-1)
            y = torch.matmul(attn, v)

            y = y.view(b, hp // ws, wp // ws, self.num_heads, ws, ws, c // self.num_heads)
            y = y.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
            y = y.view(b, c, hp, wp)

            if pad_h or pad_w:
                y = y[:, :, :h, :w]

            x = x + self.proj(y)
            x = x + self.mlp(self.norm2(x))
            return x

    def __init__(self, in_channels=1, out_channels=1, n_c=32, depth=4):
        super().__init__()

        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.attn = nn.ModuleList()
        prev_ch = in_channels

        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            self.attn.append(self.WindowAttentionBlock(ch))
            prev_ch = ch

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.WindowAttentionBlock(channels[-1]),
        )

        self.up_proj = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            self.up_proj.append(nn.Conv2d(channels[i + 1], channels[i], 1))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked

        for i, (encoder, attn) in enumerate(zip(self.encoders, self.attn)):
            y = attn(encoder(y))
            skips.append(y)
            if i < len(self.encoders) - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for up, decoder, skip in zip(self.up_proj, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(y)
            y = decoder(torch.cat([y, skip], dim=1))

        output = self.head(y)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.expand(-1, output.shape[1], -1, -1)

        output = output.clone()
        output[~valid] = float("nan")
        return output


