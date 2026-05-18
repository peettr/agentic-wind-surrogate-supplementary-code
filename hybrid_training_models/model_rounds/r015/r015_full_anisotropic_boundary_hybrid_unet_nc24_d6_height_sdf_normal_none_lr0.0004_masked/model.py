import torch
import torch.nn as nn
import torch.nn.functional as F

class anisotropic_boundary_hybrid_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=(3, 3), dilation=(1, 1), groups=1):
            super().__init__()
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(dilation, int):
                dilation = (dilation, dilation)
            pad_h = dilation[0] * (kernel_size[0] - 1) // 2
            pad_w = dilation[1] * (kernel_size[1] - 1) // 2
            self.pad = nn.ReflectionPad2d((pad_w, pad_w, pad_h, pad_h))
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, dilation=dilation, groups=groups, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.proj = None if in_ch == out_ch else anisotropic_boundary_hybrid_unet.ReflectConv(in_ch, out_ch, 1)
            self.net = nn.Sequential(
                anisotropic_boundary_hybrid_unet.ReflectConv(in_ch, out_ch, (3, 3)),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                anisotropic_boundary_hybrid_unet.ReflectConv(out_ch, out_ch, (1, 5)),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                anisotropic_boundary_hybrid_unet.ReflectConv(out_ch, out_ch, (5, 1)),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            r = x if self.proj is None else self.proj(x)
            return self.act(self.net(x) + r)

    class Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.pad = nn.ReflectionPad2d((0, 1, 0, 1))
            self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=0, bias=False)
            self.block = anisotropic_boundary_hybrid_unet.ConvBlock(out_ch, out_ch)

        def forward(self, x):
            return self.block(self.down(self.pad(x)))

    class Up(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = anisotropic_boundary_hybrid_unet.ReflectConv(in_ch + skip_ch, out_ch, 1)
            self.block = anisotropic_boundary_hybrid_unet.ConvBlock(out_ch, out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            return self.block(self.reduce(x))

    class Bottleneck(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.local = anisotropic_boundary_hybrid_unet.ConvBlock(ch, ch)
            self.dilated = nn.Sequential(
                anisotropic_boundary_hybrid_unet.ReflectConv(ch, ch, 3, dilation=(1, 2)),
                nn.GroupNorm(min(8, ch), ch),
                nn.SiLU(inplace=True),
                anisotropic_boundary_hybrid_unet.ReflectConv(ch, ch, 3, dilation=(2, 1)),
                nn.GroupNorm(min(8, ch), ch),
                nn.SiLU(inplace=True),
                anisotropic_boundary_hybrid_unet.ReflectConv(ch, ch, 3, dilation=(2, 2)),
                nn.GroupNorm(min(8, ch), ch),
            )
            self.mix = anisotropic_boundary_hybrid_unet.ReflectConv(ch, ch, 1)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.mix(self.local(x) + self.dilated(x)))

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem = self.ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList([self.Down(channels[i], channels[i + 1]) for i in range(depth - 1)])
        self.bottleneck = self.Bottleneck(channels[-1])
        self.decoder = nn.ModuleList([
            self.Up(channels[i + 1], channels[i], channels[i])
            for i in range(depth - 2, -1, -1)
        ])
        self.head = nn.Sequential(
            self.ConvBlock(channels[0], channels[0]),
            self.ReflectConv(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for down in self.encoder:
            h = down(h)
            skips.append(h)

        h = self.bottleneck(h)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            h = up(h, skip)

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] == out.shape[1]:
            valid_out = valid
        else:
            valid_out = valid[:, :1].expand_as(out)
        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


