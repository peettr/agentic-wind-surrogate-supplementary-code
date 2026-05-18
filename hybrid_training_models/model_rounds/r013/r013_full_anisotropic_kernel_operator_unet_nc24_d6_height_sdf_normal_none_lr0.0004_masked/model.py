import torch
import torch.nn as nn
import torch.nn.functional as F

class anisotropic_kernel_operator_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size):
            super().__init__()
            if isinstance(kernel_size, int):
                kh, kw = kernel_size, kernel_size
            else:
                kh, kw = kernel_size
            self.pad = nn.ReflectionPad2d((kw // 2, kw // 2, kh // 2, kh // 2))
            self.conv = nn.Conv2d(in_ch, out_ch, (kh, kw), padding=0, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class AnisoBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            groups = 8
            while out_ch % groups != 0 and groups > 1:
                groups //= 2

            self.proj = None
            if in_ch != out_ch:
                self.proj = anisotropic_kernel_operator_unet.RefConv(in_ch, out_ch, 1)

            self.conv1 = anisotropic_kernel_operator_unet.RefConv(in_ch, out_ch, 3)
            self.norm1 = nn.GroupNorm(groups, out_ch)
            self.conv_h = anisotropic_kernel_operator_unet.RefConv(out_ch, out_ch, (1, 7))
            self.conv_v = anisotropic_kernel_operator_unet.RefConv(out_ch, out_ch, (7, 1))
            self.norm2 = nn.GroupNorm(groups, out_ch)
            self.mix = anisotropic_kernel_operator_unet.RefConv(out_ch, out_ch, 1)
            self.norm3 = nn.GroupNorm(groups, out_ch)

        def forward(self, x):
            residual = x if self.proj is None else self.proj(x)
            x = F.gelu(self.norm1(self.conv1(x)))
            x = self.conv_h(x) + self.conv_v(x)
            x = F.gelu(self.norm2(x))
            x = self.norm3(self.mix(x))
            return F.gelu(x + residual)

    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.AnisoBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.AnisoBlock(channels[-1], channels[-1])

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(self.RefConv(channels[i + 1], channels[i], 1))
            self.decoders.append(self.AnisoBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.RefConv(channels[0], channels[0], 3),
            nn.GELU(),
            self.RefConv(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for i, enc in enumerate(self.encoders):
            y = enc(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels == x.shape[1]:
            out_valid = valid
        else:
            out_valid = valid.any(dim=1, keepdim=True).expand_as(y)

        y = torch.where(out_valid, y, torch.full_like(y, float("nan")))
        return y


