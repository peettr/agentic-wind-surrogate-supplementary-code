import torch
import torch.nn as nn
import torch.nn.functional as F


class tt_adapter_decoder_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.depth = depth
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.out(h)

        output_mask = valid
        if output_mask.shape[1] != output.shape[1]:
            output_mask = output_mask.all(dim=1, keepdim=True)

        output = torch.where(output_mask, output, torch.full_like(output, float("nan")))
        return output