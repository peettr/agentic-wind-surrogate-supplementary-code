import torch
import torch.nn as nn
import torch.nn.functional as F

class hypernet_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=5):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self._block(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        for i in range(1, depth):
            self.encoders.append(self._block(channels[i - 1], channels[i]))

        self.bottleneck = self._block(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoders.append(self._block(channels[i] * 2, channels[i]))

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0)

    def _conv3x3(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            self._conv3x3(in_ch, out_ch),
            self._conv3x3(out_ch, out_ch),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for enc in self.encoders:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            h = enc(h)
            skips.append(h)

        h = self.bottleneck(h)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        output = self.out_conv(self.out_pad(h))
        output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.expand(-1, output.shape[1], -1, -1)

        output = output.clone()
        output[~valid] = float("nan")
        return output


