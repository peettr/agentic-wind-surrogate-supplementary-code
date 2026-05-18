import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = min(8, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class _ReflectConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=0, bias=bias)

    def forward(self, x):
        if x.shape[-2] <= 1 or x.shape[-1] <= 1:
            x = F.interpolate(x, size=(max(2, x.shape[-2]), max(2, x.shape[-1])), mode="nearest")
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            _ReflectConv2d(c_in, c_out, kernel_size=3, bias=False),
            _gn(c_out),
            nn.SiLU(inplace=True),
            _ReflectConv2d(c_out, c_out, kernel_size=3, bias=False),
            _gn(c_out),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class dynamic_roughness_dispatch_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev_c = in_channels
        for c in channels:
            self.encoders.append(ConvBlock(prev_c, c))
            prev_c = c

        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(
                nn.Sequential(
                    _ReflectConv2d(channels[i + 1], channels[i], kernel_size=3, bias=False),
                    _gn(channels[i]),
                    nn.SiLU(inplace=True),
                )
            )
            self.decoders.append(ConvBlock(channels[i] * 2, channels[i]))

        self.out_head = nn.Sequential(
            _ReflectConv2d(channels[0], channels[0], kernel_size=3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            if i < self.depth - 1 and h.shape[-2] >= 2 and h.shape[-1] >= 2:
                skips.append(h)
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up_conv, decoder, skip in zip(self.up_convs[-len(skips):], self.decoders[-len(skips):], reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_conv(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        output = self.out_head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid.expand(-1, output.shape[1], -1, -1)
        output = output.masked_fill(~valid_out, float("nan"))
        return output


