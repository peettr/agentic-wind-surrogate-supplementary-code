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
        if depth < 1:
            raise ValueError("depth must be >= 1")

        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.encoders.append(self.ConvBlock(in_channels, channels[0]))
        for i in range(1, depth):
            self.encoders.append(self.ConvBlock(channels[i - 1], channels[i]))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = channels[i] if i == depth - 1 else channels[i + 1]
            self.up_convs.append(nn.Conv2d(in_ch, channels[i], kernel_size=1))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.final_pad = nn.ReflectionPad2d(1)
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3)

    def forward(self, x):
        valid = ~torch.isnan(x).any(dim=1, keepdim=True)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_conv(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.final_conv(self.final_pad(h))
        output = torch.where(valid.expand_as(output), output, torch.full_like(output, float("nan")))
        return output


