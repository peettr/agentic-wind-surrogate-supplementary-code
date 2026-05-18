import torch
import torch.nn as nn
import torch.nn.functional as F

class liif_residual_decoder_unet(nn.Module):
    class _ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class _Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                liif_residual_decoder_unet._ReflectConv(in_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                liif_residual_decoder_unet._ReflectConv(out_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
            self.skip = nn.Identity() if in_ch == out_ch else liif_residual_decoder_unet._ReflectConv(in_ch, out_ch, 1)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self._Block(in_channels, channels[0])
        self.encoder = nn.ModuleList()
        for i in range(depth - 1):
            self.encoder.append(self._Block(channels[i], channels[i + 1]))

        self.bottleneck = nn.Sequential(
            self._Block(channels[-1], channels[-1]),
            self._Block(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(self._Block(channels[i + 1], channels[i]))
            self.fuse.append(self._Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self._Block(channels[0], channels[0]),
            self._ReflectConv(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for block in self.encoder:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            h = block(h)
            skips.append(h)

        h = self.bottleneck(h)

        for block, fuse, skip in zip(self.decoder, self.fuse, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = block(h)
            h = fuse(torch.cat([h, skip], dim=1))

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != output.shape[1]:
            valid_out = valid_out[:, :1].expand_as(output)

        output = output.clone()
        output[~valid_out] = float("nan")
        return output