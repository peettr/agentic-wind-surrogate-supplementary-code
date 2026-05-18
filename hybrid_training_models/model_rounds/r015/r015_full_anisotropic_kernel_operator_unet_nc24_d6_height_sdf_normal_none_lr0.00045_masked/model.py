import torch
import torch.nn as nn
import torch.nn.functional as F

class anisotropic_kernel_operator_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
            super().__init__()
            if isinstance(kernel_size, int):
                pad = kernel_size // 2
                self.pad = nn.ReflectionPad2d(pad)
            else:
                kh, kw = kernel_size
                self.pad = nn.ReflectionPad2d((kw // 2, kw // 2, kh // 2, kh // 2))
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=bias,
            )

        def forward(self, x):
            return self.conv(self.pad(x))

    class Block(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.in_proj = anisotropic_kernel_operator_unet.ReflectConv(in_channels, out_channels, 3)
            self.vert = anisotropic_kernel_operator_unet.ReflectConv(out_channels, out_channels, (7, 3))
            self.horz = anisotropic_kernel_operator_unet.ReflectConv(out_channels, out_channels, (3, 7))
            self.mix = anisotropic_kernel_operator_unet.ReflectConv(out_channels * 2, out_channels, 1)
            self.out = anisotropic_kernel_operator_unet.ReflectConv(out_channels, out_channels, 3)
            self.skip = (
                anisotropic_kernel_operator_unet.ReflectConv(in_channels, out_channels, 1)
                if in_channels != out_channels
                else nn.Identity()
            )
            self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
            self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

        def forward(self, x):
            residual = self.skip(x)
            x = F.silu(self.norm1(self.in_proj(x)))
            a = self.vert(x)
            b = self.horz(x)
            x = F.silu(self.mix(torch.cat([a, b], dim=1)))
            x = self.norm2(self.out(x))
            return F.silu(x + residual)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = max(1, depth)

        channels = [n_c * min(2 ** i, 8) for i in range(self.depth)]

        self.stem = self.Block(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(1, self.depth):
            self.down.append(self.ReflectConv(channels[i - 1], channels[i], 3, stride=2))
            self.encoder.append(self.Block(channels[i], channels[i]))

        self.bottleneck = nn.Sequential(
            self.Block(channels[-1], channels[-1]),
            self.Block(channels[-1], channels[-1]),
        )

        self.up_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(self.depth - 2, -1, -1):
            self.up_proj.append(self.ReflectConv(channels[i + 1], channels[i], 1))
            self.decoder.append(self.Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.Block(channels[0], channels[0]),
            self.ReflectConv(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        x = self.stem(x_masked)
        skips.append(x)

        for down, block in zip(self.down, self.encoder):
            x = F.silu(down(x))
            x = block(x)
            skips.append(x)

        x = self.bottleneck(x)

        for proj, block, skip in zip(self.up_proj, self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = proj(x)
            x = block(torch.cat([x, skip], dim=1))

        x = self.head(x)
        if x.shape[-2:] != x_masked.shape[-2:]:
            x = F.interpolate(x, size=x_masked.shape[-2:], mode="bilinear", align_corners=False)

        output_valid = valid
        if output_valid.shape[1] != x.shape[1]:
            output_valid = output_valid[:, :1].expand(-1, x.shape[1], -1, -1)

        return torch.where(output_valid, x, torch.full_like(x, float("nan")))