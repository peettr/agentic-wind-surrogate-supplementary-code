import torch
import torch.nn as nn
import torch.nn.functional as F

class p8_dcn_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [n_c * (2 ** i) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoder.append(self.ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        prev_channels = channels[-1] * 2
        for ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(prev_channels, ch, kernel_size=2, stride=2))
            self.decoder.append(self.ConvBlock(ch * 2, ch))
            prev_channels = ch

        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(0),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for enc in self.encoder:
            h = enc(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([skip, h], dim=1)
            h = dec(h)

        output = self.final(h)

        valid = valid.expand_as(output)
        output = output.clone()
        output[~valid] = float("nan")
        return output


