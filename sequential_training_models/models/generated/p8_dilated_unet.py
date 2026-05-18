import torch
import torch.nn as nn
import torch.nn.functional as F

class p8_dilated_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, dilation):
            super().__init__()
            pad = dilation
            self.block = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, dilation=dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(pad),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, dilation=dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7, dilation=2):
        super().__init__()
        self.depth = depth

        channels = [n_c * (2 ** i) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder.append(self.ConvBlock(prev_ch, ch, dilation))
            prev_ch = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[i + 1], channels[i], kernel_size=2, stride=2))
            self.decoder.append(self.ConvBlock(channels[i] * 2, channels[i], dilation))

        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x).any(dim=1, keepdim=True)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, enc in enumerate(self.encoder):
            h = enc(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.pool(h)

        skips = skips[:-1][::-1]

        for up, dec, skip in zip(self.upconvs, self.decoder, skips):
            h = up(h)

            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            h = torch.cat([skip, h], dim=1)
            h = dec(h)

        output = self.out_conv(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid = valid.expand(-1, output.shape[1], -1, -1)
        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output