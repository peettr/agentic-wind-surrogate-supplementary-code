import torch
import torch.nn as nn
import torch.nn.functional as F

class anisotropic_kernel_operator_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size):
            super().__init__()
            if isinstance(kernel_size, int):
                kh, kw = kernel_size, kernel_size
            else:
                kh, kw = kernel_size
            self.pad = nn.ReflectionPad2d((kw // 2, kw // 2, kh // 2, kh // 2))
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw), padding=0)

        def forward(self, x):
            return self.conv(self.pad(x))

    class AKOBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
            self.conv_in = anisotropic_kernel_operator_unet.ReflectConv(in_ch, out_ch, 3)
            self.conv_x = anisotropic_kernel_operator_unet.ReflectConv(out_ch, out_ch, (1, 7))
            self.conv_y = anisotropic_kernel_operator_unet.ReflectConv(out_ch, out_ch, (7, 1))
            self.conv_mix = anisotropic_kernel_operator_unet.ReflectConv(out_ch, out_ch, 3)
            self.norm1 = nn.BatchNorm2d(out_ch)
            self.norm2 = nn.BatchNorm2d(out_ch)
            self.norm3 = nn.BatchNorm2d(out_ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            skip = self.proj(x)
            y = self.act(self.norm1(self.conv_in(x)))
            y = self.act(self.norm2(self.conv_x(y) + self.conv_y(y)))
            y = self.norm3(self.conv_mix(y))
            return self.act(y + skip)

    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self.AKOBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            self.AKOBlock(channels[-1], channels[-1]),
            self.AKOBlock(channels[-1], channels[-1])
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoders.append(self.AKOBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked
        for i, enc in enumerate(self.encoders):
            y = enc(y)
            skips.append(y)
            if i != len(self.encoders) - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(y)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False) if y.shape[-2:] != x.shape[-2:] else y

        out_valid = valid
        if out_valid.shape[1] != y.shape[1]:
            out_valid = out_valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)
        return torch.where(out_valid, y, torch.full_like(y, float("nan")))


