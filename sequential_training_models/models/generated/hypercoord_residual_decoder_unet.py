import torch
import torch.nn as nn
import torch.nn.functional as F

class hypercoord_residual_decoder_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.max_channels = n_c * 8

        channels = [min(n_c * (2 ** i), self.max_channels) for i in range(depth)]
        self.channels = channels

        class ReflectConv(nn.Module):
            def __init__(self, c_in, c_out, k=3):
                super().__init__()
                self.pad = nn.ReflectionPad2d(k // 2)
                self.conv = nn.Conv2d(c_in, c_out, k, padding=0, bias=False)

            def forward(self, x):
                return self.conv(self.pad(x))

        class ResBlock(nn.Module):
            def __init__(self, c_in, c_out):
                super().__init__()
                self.conv1 = ReflectConv(c_in, c_out)
                self.norm1 = nn.GroupNorm(min(8, c_out), c_out)
                self.conv2 = ReflectConv(c_out, c_out)
                self.norm2 = nn.GroupNorm(min(8, c_out), c_out)
                self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)

            def forward(self, x):
                y = F.silu(self.norm1(self.conv1(x)))
                y = self.norm2(self.conv2(y))
                return F.silu(y + self.skip(x))

        self.stem = ResBlock(in_channels + 2, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(ResBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            ResBlock(channels[-1], channels[-1]),
            ResBlock(channels[-1], channels[-1])
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(ResBlock(channels[i + 1] + channels[i] + 2, channels[i]))

        self.head = nn.Sequential(
            ResBlock(channels[0] + 2, channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0)
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
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        coords = self._coords(x_masked)
        y = self.stem(torch.cat([x_masked, coords], dim=1))

        skips = [y]
        for block in self.encoder:
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            coords_y = self._coords(y)
            y = block(torch.cat([y, skip, coords_y], dim=1))

        y = torch.cat([y, self._coords(y)], dim=1)
        y = self.head(y)

        if self.in_channels == self.out_channels:
            y = y + x_masked

        return torch.where(valid, y, torch.full_like(y, float("nan")))