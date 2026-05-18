import torch
import torch.nn as nn
import torch.nn.functional as F

class one_sided_lora_output_adapter_unet(nn.Module):
    class ReflectionConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                one_sided_lora_output_adapter_unet.ReflectionConv(in_channels, out_channels, 3),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                one_sided_lora_output_adapter_unet.ReflectionConv(out_channels, out_channels, 3),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UpBlock(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.reduce = one_sided_lora_output_adapter_unet.ReflectionConv(in_channels, out_channels, 3)
            self.block = one_sided_lora_output_adapter_unet.ConvBlock(out_channels + skip_channels, out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            return self.block(x)

    class LoRAOutputAdapter(nn.Module):
        def __init__(self, in_channels, out_channels, rank=8):
            super().__init__()
            rank = max(1, min(rank, in_channels, 32))
            self.base = one_sided_lora_output_adapter_unet.ReflectionConv(in_channels, out_channels, 3)
            self.down = nn.Conv2d(in_channels, rank, kernel_size=1, padding=0, bias=False)
            self.up = nn.Conv2d(rank, out_channels, kernel_size=1, padding=0, bias=False)
            nn.init.zeros_(self.up.weight)

        def forward(self, x):
            return self.base(x) + self.up(self.down(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        decoder_in = channels[-1]
        for skip_ch in reversed(channels):
            self.decoders.append(self.UpBlock(decoder_in, skip_ch, skip_ch))
            decoder_in = skip_ch

        self.output_adapter = self.LoRAOutputAdapter(channels[0], out_channels, rank=8)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            h = decoder(h, skip)

        out = self.output_adapter(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid_out = valid[:, :1].expand(-1, out.shape[1], -1, -1)
        else:
            valid_out = valid

        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


