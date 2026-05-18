import torch
import torch.nn as nn
import torch.nn.functional as F

class multiscale_linear_attention_skip_mixer_unet(nn.Module):
    class RefConv2d(nn.Module):
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
            groups = min(8, out_ch)
            while out_ch % groups != 0:
                groups -= 1
            self.net = nn.Sequential(
                multiscale_linear_attention_skip_mixer_unet.RefConv2d(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
                multiscale_linear_attention_skip_mixer_unet.RefConv2d(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
            )
            self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

        def forward(self, x):
            return self.net(x) + self.proj(x)

    class LinearAttention(nn.Module):
        def __init__(self, ch, heads=4):
            super().__init__()
            heads = min(heads, max(1, ch // 16))
            while ch % heads != 0:
                heads -= 1
            self.heads = heads
            self.dim_head = ch // heads
            self.to_qkv = nn.Conv2d(ch, ch * 3, 1, bias=False)
            self.out = nn.Conv2d(ch, ch, 1, bias=False)
            self.scale = self.dim_head ** -0.5

        def forward(self, x):
            b, c, h, w = x.shape
            q, k, v = self.to_qkv(x).chunk(3, dim=1)
            q = q.view(b, self.heads, self.dim_head, h * w)
            k = k.view(b, self.heads, self.dim_head, h * w)
            v = v.view(b, self.heads, self.dim_head, h * w)
            q = F.softmax(q, dim=2) * self.scale
            k = F.softmax(k, dim=3)
            context = torch.matmul(k, v.transpose(-1, -2))
            y = torch.matmul(context.transpose(-1, -2), q).view(b, c, h, w)
            return x + self.out(y)

    class SkipMixer(nn.Module):
        def __init__(self, skip_ch, dec_ch, out_ch):
            super().__init__()
            self.skip_gate = nn.Sequential(
                nn.Conv2d(skip_ch + dec_ch, out_ch, 1, bias=True),
                nn.Sigmoid(),
            )
            self.skip_proj = nn.Conv2d(skip_ch, out_ch, 1, bias=False)
            self.dec_proj = nn.Conv2d(dec_ch, out_ch, 1, bias=False)
            self.refine = multiscale_linear_attention_skip_mixer_unet.ConvBlock(out_ch, out_ch)

        def forward(self, dec, skip):
            if dec.shape[-2:] != skip.shape[-2:]:
                dec = F.interpolate(dec, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            gate = self.skip_gate(torch.cat([skip, dec], dim=1))
            x = self.dec_proj(dec) + gate * self.skip_proj(skip)
            return self.refine(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self.ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(1, depth):
            self.down.append(nn.AvgPool2d(2))
            self.encoder.append(self.ConvBlock(channels[i - 1], channels[i]))

        self.attn_levels = nn.ModuleList([
            self.LinearAttention(ch) if i >= max(1, depth - 3) else nn.Identity()
            for i, ch in enumerate(channels)
        ])

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.LinearAttention(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self.SkipMixer(channels[i], channels[i + 1], channels[i]))

        self.head = nn.Sequential(
            self.ConvBlock(channels[0], channels[0]),
            self.RefConv2d(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        h = self.attn_levels[0](h)
        skips.append(h)

        for i, (down, enc) in enumerate(zip(self.down, self.encoder), start=1):
            h = down(h)
            h = enc(h)
            h = self.attn_levels[i](h)
            skips.append(h)

        h = self.bottleneck(skips[-1])

        for dec, skip in zip(self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            h = dec(h, skip)

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid_out = valid.all(dim=1, keepdim=True).expand_as(out)
        else:
            valid_out = valid
        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


