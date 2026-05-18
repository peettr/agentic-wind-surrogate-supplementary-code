import torch
import torch.nn as nn
import torch.nn.functional as F

class coord_field_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels + 2
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(channels[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, ch, kernel_size=2, stride=2))
            self.decoders.append(self.ConvBlock(ch + ch, ch))
            prev_ch = ch

        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def _coords(self, x):
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        yy = yy.expand(b, 1, h, w)
        xx = xx.expand(b, 1, h, w)
        return torch.cat([xx, yy], dim=1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        y = torch.cat([x_masked, self._coords(x_masked)], dim=1)

        skips = []
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.pool(y)

        y = self.bottleneck(y)

        skips = skips[:-1][::-1]
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, skips):
            y = upconv(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = decoder(y)

        y = self.final(y)

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(y)

        y = y.masked_fill(~valid, float("nan"))
        return y


