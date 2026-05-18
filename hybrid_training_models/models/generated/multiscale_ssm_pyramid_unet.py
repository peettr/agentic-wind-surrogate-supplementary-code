import torch
import torch.nn as nn
import torch.nn.functional as F

class multiscale_ssm_pyramid_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, groups=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, groups=groups, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)
            self.conv1 = multiscale_ssm_pyramid_unet.ReflectConv(in_ch, out_ch, 3, bias=False)
            self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.conv2 = multiscale_ssm_pyramid_unet.ReflectConv(out_ch, out_ch, 3, groups=out_ch, bias=False)
            self.pw2 = nn.Conv2d(out_ch, out_ch, 1, padding=0, bias=False)
            self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            r = self.proj(x)
            x = self.act(self.norm1(self.conv1(x)))
            x = self.norm2(self.pw2(self.conv2(x)))
            return self.act(x + r)

    class AxialSSMBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.norm = nn.GroupNorm(min(8, ch), ch)
            self.local = multiscale_ssm_pyramid_unet.ReflectConv(ch, ch, 3, groups=ch, bias=False)
            self.h_gate = nn.Conv2d(ch, ch, 1, padding=0)
            self.w_gate = nn.Conv2d(ch, ch, 1, padding=0)
            self.mix = nn.Conv2d(ch * 3, ch, 1, padding=0, bias=False)
            self.out = nn.Conv2d(ch, ch, 1, padding=0, bias=False)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            r = x
            z = self.norm(x)
            local = self.local(z)

            h = z.mean(dim=3, keepdim=True)
            h = torch.cumsum(h, dim=2)
            h = h / torch.arange(1, h.shape[2] + 1, device=h.device, dtype=h.dtype).view(1, 1, -1, 1)
            h = torch.sigmoid(self.h_gate(h)).expand_as(z) * z

            w = z.mean(dim=2, keepdim=True)
            w = torch.cumsum(w, dim=3)
            w = w / torch.arange(1, w.shape[3] + 1, device=w.device, dtype=w.dtype).view(1, 1, 1, -1)
            w = torch.sigmoid(self.w_gate(w)).expand_as(z) * z

            z = self.mix(torch.cat([local, h, w], dim=1))
            z = self.out(self.act(z))
            return r + z

    class DownBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = multiscale_ssm_pyramid_unet.ConvBlock(in_ch, out_ch)
            self.ssm = multiscale_ssm_pyramid_unet.AxialSSMBlock(out_ch)

        def forward(self, x):
            x = self.block(x)
            x = self.ssm(x)
            return x

    class UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)
            self.block = multiscale_ssm_pyramid_unet.ConvBlock(out_ch + skip_ch, out_ch)
            self.ssm = multiscale_ssm_pyramid_unet.AxialSSMBlock(out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            x = self.block(x)
            x = self.ssm(x)
            return x

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self.ConvBlock(in_channels, channels[0])

        enc = []
        prev = channels[0]
        for ch in channels:
            enc.append(self.DownBlock(prev, ch))
            prev = ch
        self.encoder = nn.ModuleList(enc)

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.AxialSSMBlock(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        dec = []
        in_ch = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            dec.append(self.UpBlock(in_ch, skip_ch, skip_ch))
            in_ch = skip_ch
        self.decoder = nn.ModuleList(dec)

        self.head = nn.Sequential(
            self.ConvBlock(channels[0], channels[0]),
            self.ReflectConv(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x = torch.where(valid, x, torch.zeros_like(x))

        x = self.stem(x)
        skips = []

        for i, block in enumerate(self.encoder):
            x = block(x)
            skips.append(x)
            if i != len(self.encoder) - 1:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = block(x, skip)

        x = self.head(x)

        if x.shape[-2:] != valid.shape[-2:]:
            x = F.interpolate(x, size=valid.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] == x.shape[1]:
            restore_mask = valid
        else:
            restore_mask = valid.all(dim=1, keepdim=True).expand(-1, x.shape[1], -1, -1)

        x = x.masked_fill(~restore_mask, float("nan"))
        return x