import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    for groups in range(min(8, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            ReflectionConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(_gn(out_channels), out_channels),
            nn.SiLU(inplace=True),
            ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(_gn(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class colora_decoder_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        decoder_in = channels[-1]
        for skip_ch in reversed(channels[:-1]):
            self.decoders.append(UpBlock(decoder_in, skip_ch, skip_ch))
            decoder_in = skip_ch

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i < len(self.encoders) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            h = decoder(h, skip)

        output = self.out_conv(self.out_pad(h))

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(output)

        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output