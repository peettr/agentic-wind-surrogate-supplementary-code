import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels):
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class conditional_basis_decoder_mixer_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=bias,
            )

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                conditional_basis_decoder_mixer_unet.ReflectConv(in_channels, out_channels, 3, bias=False),
                _gn(out_channels),
                nn.SiLU(inplace=True),
                conditional_basis_decoder_mixer_unet.ReflectConv(out_channels, out_channels, 3, bias=False),
                _gn(out_channels),
                nn.SiLU(inplace=True),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
            )

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class BasisMixer(nn.Module):
        def __init__(self, channels, num_basis=8):
            super().__init__()
            hidden = max(channels // 4, 8)
            self.basis = nn.Parameter(torch.randn(1, num_basis, channels, 1, 1) * 0.02)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, num_basis, kernel_size=1),
            )
            self.proj = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)

        def forward(self, x):
            w = torch.softmax(self.gate(x), dim=1).unsqueeze(2)
            scale = (w * self.basis).sum(dim=1)
            return x + self.proj(x * (1.0 + scale))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_proj = (
            nn.Identity()
            if in_channels == in_channels
            else None
        )

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev, ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.BasisMixer(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.skip_mixers = nn.ModuleList([self.BasisMixer(ch) for ch in channels])

        self.decoders = nn.ModuleList()
        dec_in = channels[-1]
        for ch in reversed(channels):
            self.decoders.append(self.ConvBlock(dec_in + ch, ch))
            dec_in = ch

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0),
        )

    def _adapt_input(self, x):
        expected = self.in_channels
        actual = x.shape[1]
        if actual == expected:
            return x
        if actual > expected:
            if actual % expected == 0:
                groups = actual // expected
                b, c, h, w = x.shape
                return x.view(b, expected, groups, h, w).mean(dim=2)
            return x[:, :expected]
        repeats = (expected + actual - 1) // actual
        x = x.repeat(1, repeats, 1, 1)
        return x[:, :expected]

    def forward(self, x):
        x = self._adapt_input(x)

        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for decoder, skip, mixer in zip(self.decoders, reversed(skips), reversed(self.skip_mixers)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = mixer(skip)
            h = decoder(torch.cat([h, skip], dim=1))

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_all = valid.all(dim=1, keepdim=True)
        valid_out = valid_all.expand(-1, out.shape[1], -1, -1)

        return torch.where(valid_out, out, torch.full_like(out, float("nan")))