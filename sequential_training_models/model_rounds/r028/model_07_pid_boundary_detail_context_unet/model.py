import torch
import torch.nn as nn
import torch.nn.functional as F

class pid_boundary_detail_context_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self._block(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            self._block(channels[-1], channels[-1]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[-1], channels[-1], kernel_size=3, bias=False),
            nn.GroupNorm(self._groups(channels[-1]), channels[-1]),
            nn.SiLU(inplace=True),
        )

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            in_ch = channels[i + 1]
            skip_ch = channels[i]
            self.up_convs.append(nn.Conv2d(in_ch, skip_ch, kernel_size=1, bias=False))
            self.decoders.append(self._block(skip_ch * 2, skip_ch))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, bias=False),
            nn.GroupNorm(self._groups(channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def _groups(self, channels):
        for g in (8, 6, 4, 3, 2):
            if channels % g == 0:
                return g
        return 1

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=False),
            nn.GroupNorm(self._groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, bias=False),
            nn.GroupNorm(self._groups(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i != self.depth - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for up_conv, decoder, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_conv(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape != output.shape:
            valid = valid.all(dim=1, keepdim=True).expand_as(output)

        output = output.clone()
        output[~valid] = float("nan")
        return output


