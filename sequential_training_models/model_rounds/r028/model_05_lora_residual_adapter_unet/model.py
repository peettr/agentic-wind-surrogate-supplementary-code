import torch
import torch.nn as nn
import torch.nn.functional as F

class lora_residual_adapter_unet(nn.Module):
    class RefConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0)

        def forward(self, x):
            return self.conv(self.pad(x))

    class LoRAAdapter(nn.Module):
        def __init__(self, channels, rank=None):
            super().__init__()
            if rank is None:
                rank = max(4, channels // 8)
            self.down = nn.Conv2d(channels, rank, 1, padding=0, bias=False)
            self.up = nn.Conv2d(rank, channels, 1, padding=0, bias=False)
            nn.init.kaiming_normal_(self.down.weight, nonlinearity="linear")
            nn.init.zeros_(self.up.weight)

        def forward(self, x):
            return self.up(self.down(x))

    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            groups = min(8, out_ch)
            while out_ch % groups != 0:
                groups -= 1

            self.conv1 = lora_residual_adapter_unet.RefConv2d(in_ch, out_ch, 3)
            self.norm1 = nn.GroupNorm(groups, out_ch)
            self.conv2 = lora_residual_adapter_unet.RefConv2d(out_ch, out_ch, 3)
            self.norm2 = nn.GroupNorm(groups, out_ch)
            self.adapter = lora_residual_adapter_unet.LoRAAdapter(out_ch)

            if in_ch == out_ch:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Conv2d(in_ch, out_ch, 1, padding=0)

        def forward(self, x):
            residual = self.skip(x)
            x = F.silu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            x = x + self.adapter(x)
            return F.silu(x + residual)

    def __init__(self, in_channels=1, out_channels=1, n_c=32, depth=6):
        super().__init__()
        max_channels = n_c * 8
        self.depth = depth

        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ResBlock(prev_ch, ch))
            prev_ch = ch

        self.down = nn.AvgPool2d(2)

        self.bottleneck = nn.Sequential(
            self.ResBlock(channels[-1], channels[-1]),
            self.ResBlock(channels[-1], channels[-1])
        )

        self.decoders = nn.ModuleList()
        decoder_in = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.decoders.append(self.ResBlock(decoder_in + skip_ch, skip_ch))
            decoder_in = skip_ch

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.out_conv(self.out_pad(h))

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output



