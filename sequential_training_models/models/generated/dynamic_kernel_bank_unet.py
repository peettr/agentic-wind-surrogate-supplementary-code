import torch
import torch.nn as nn
import torch.nn.functional as F


class dynamic_kernel_bank_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=bias,
            )

        def forward(self, x):
            return self.conv(self.pad(x))

    class SEGate(nn.Module):
        def __init__(self, channels, reduction=8):
            super().__init__()
            hidden = max(channels // reduction, 4)
            self.fc1 = nn.Conv2d(channels, hidden, 1)
            self.fc2 = nn.Conv2d(hidden, channels, 1)

        def forward(self, x):
            w = F.adaptive_avg_pool2d(x, 1)
            w = F.silu(self.fc1(w))
            w = torch.sigmoid(self.fc2(w))
            return x * w

    class DynamicKernelBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

            self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.conv3 = dynamic_kernel_bank_unet.ReflectConv(out_ch, out_ch, 3, groups=out_ch, bias=False)
            self.conv5 = dynamic_kernel_bank_unet.ReflectConv(out_ch, out_ch, 5, groups=out_ch, bias=False)
            self.conv7 = dynamic_kernel_bank_unet.ReflectConv(out_ch, out_ch, 7, groups=out_ch, bias=False)

            hidden = max(out_ch // 4, 8)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, hidden, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 3, 1),
            )

            self.mix = nn.Conv2d(out_ch, out_ch, 1, bias=False)
            self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.ffn1 = nn.Conv2d(out_ch, out_ch * 2, 1)
            self.ffn2 = nn.Conv2d(out_ch * 2, out_ch, 1)
            self.se = dynamic_kernel_bank_unet.SEGate(out_ch)

        def forward(self, x):
            x = self.proj(x)
            residual = x

            y = self.norm1(x)
            weights = torch.softmax(self.gate(y), dim=1)
            y = (
                self.conv3(y) * weights[:, 0:1]
                + self.conv5(y) * weights[:, 1:2]
                + self.conv7(y) * weights[:, 2:3]
            )
            y = self.mix(y)
            x = residual + y

            residual = x
            y = self.norm2(x)
            y = self.ffn2(F.silu(self.ffn1(y)))
            y = self.se(y)
            return residual + y

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.ReflectConv(in_channels, channels[0], 3, bias=False)

        self.encoder = nn.ModuleList()
        for i in range(depth):
            out_ch = channels[i]
            self.encoder.append(
                nn.Sequential(
                    self.DynamicKernelBlock(out_ch, out_ch),
                    self.DynamicKernelBlock(out_ch, out_ch),
                )
            )

        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AvgPool2d(2),
                    self.ReflectConv(channels[i], channels[i + 1], 3, bias=False),
                )
                for i in range(depth - 1)
            ]
        )

        self.bottleneck = nn.Sequential(
            self.DynamicKernelBlock(channels[-1], channels[-1]),
            self.DynamicKernelBlock(channels[-1], channels[-1]),
        )

        self.up_proj = nn.ModuleList(
            [
                nn.Conv2d(channels[i + 1], channels[i], 1, bias=False)
                for i in reversed(range(depth - 1))
            ]
        )

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    self.DynamicKernelBlock(channels[i] * 2, channels[i]),
                    self.DynamicKernelBlock(channels[i], channels[i]),
                )
                for i in reversed(range(depth - 1))
            ]
        )

        self.head = nn.Sequential(
            self.DynamicKernelBlock(channels[0], channels[0]),
            self.ReflectConv(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        y = self.stem(x_masked)
        skips = []

        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.down[i](y)

        y = self.bottleneck(y)

        for up, block, skip in zip(self.up_proj, self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(y)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        output = self.head(y)
        output = output.masked_fill(~valid.expand_as(output), float("nan"))
        return output



