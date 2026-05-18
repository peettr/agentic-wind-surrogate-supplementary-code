import torch
import torch.nn as nn
import torch.nn.functional as F

class height_only_anisotropic_compact_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, groups=1):
            super().__init__()
            if isinstance(kernel_size, int):
                kh, kw = kernel_size, kernel_size
            else:
                kh, kw = kernel_size
            self.pad = nn.ReflectionPad2d((kw // 2, kw // 2, kh // 2, kh // 2))
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw), padding=0, groups=groups, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class AnisotropicBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.proj = None
            if in_ch != out_ch:
                self.proj = height_only_anisotropic_compact_unet.ReflectConv(in_ch, out_ch, 1)

            self.conv_in = height_only_anisotropic_compact_unet.ReflectConv(in_ch, out_ch, 3)
            self.norm_in = nn.GroupNorm(min(8, out_ch), out_ch)

            self.conv_h = height_only_anisotropic_compact_unet.ReflectConv(out_ch, out_ch, (1, 5), groups=1)
            self.conv_v = height_only_anisotropic_compact_unet.ReflectConv(out_ch, out_ch, (5, 1), groups=1)
            self.norm_mid = nn.GroupNorm(min(8, out_ch), out_ch)

            self.conv_out = height_only_anisotropic_compact_unet.ReflectConv(out_ch, out_ch, 3)
            self.norm_out = nn.GroupNorm(min(8, out_ch), out_ch)

        def forward(self, x):
            residual = x if self.proj is None else self.proj(x)

            y = self.conv_in(x)
            y = self.norm_in(y)
            y = F.gelu(y)

            y = self.conv_h(y) + self.conv_v(y)
            y = self.norm_mid(y)
            y = F.gelu(y)

            y = self.conv_out(y)
            y = self.norm_out(y)

            return F.gelu(y + residual)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(self.AnisotropicBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.ModuleList([
            nn.AvgPool2d(kernel_size=2, stride=2)
            for _ in range(depth - 1)
        ])

        self.bottleneck = nn.Sequential(
            self.AnisotropicBlock(channels[-1], channels[-1]),
            self.AnisotropicBlock(channels[-1], channels[-1])
        )

        self.up_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_proj.append(self.ReflectConv(channels[i + 1], channels[i], 1))
            self.decoder.append(self.AnisotropicBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.AnisotropicBlock(channels[0], channels[0]),
            self.ReflectConv(channels[0], out_channels, 1)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.down[i](y)

        y = self.bottleneck(y)

        for i, block in enumerate(self.decoder):
            skip = skips[-(i + 2)]
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = self.up_proj[i](y)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        y = self.head(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if y.shape[1] == valid.shape[1]:
            valid_out = valid
        else:
            valid_out = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)

        return torch.where(valid_out, y, torch.full_like(y, float("nan")))


