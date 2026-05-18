import torch
import torch.nn as nn
import torch.nn.functional as F

class dct_spectral_unet(nn.Module):
    class ReflectionConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                dct_spectral_unet.ReflectionConv2d(in_channels, out_channels, 3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
                dct_spectral_unet.ReflectionConv2d(out_channels, out_channels, 3, bias=False),
                nn.GroupNorm(min(8, out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class SpectralBlock(nn.Module):
        def __init__(self, channels, modes_h=32, modes_w=32):
            super().__init__()
            self.modes_h = modes_h
            self.modes_w = modes_w
            scale = 1.0 / max(1, channels)
            self.weight_real = nn.Parameter(scale * torch.randn(channels, modes_h, modes_w))
            self.weight_imag = nn.Parameter(scale * torch.randn(channels, modes_h, modes_w))
            self.mix = dct_spectral_unet.ReflectionConv2d(channels, channels, 1, bias=False)
            self.norm = nn.GroupNorm(min(8, channels), channels)

        def forward(self, x):
            b, c, h, w = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros_like(x_ft)

            mh = min(self.modes_h, h)
            mw = min(self.modes_w, x_ft.shape[-1])
            weight = torch.complex(
                self.weight_real[:, :mh, :mw],
                self.weight_imag[:, :mh, :mw],
            )

            out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight.unsqueeze(0)
            y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
            return F.silu(self.norm(self.mix(x) + y), inplace=True)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_channels, ch))
            prev_channels = ch

        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            self.ConvBlock(bottleneck_channels, bottleneck_channels),
            self.SpectralBlock(bottleneck_channels, modes_h=32, modes_w=32),
            self.ConvBlock(bottleneck_channels, bottleneck_channels),
        )

        self.decoders = nn.ModuleList()
        self.reduce = nn.ModuleList()
        for ch in reversed(channels[:-1]):
            self.reduce.append(self.ReflectionConv2d(prev_channels, ch, 1, bias=False))
            self.decoders.append(self.ConvBlock(ch * 2, ch))
            prev_channels = ch

        self.out_head = nn.Sequential(
            self.ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            self.ReflectionConv2d(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        skips = skips[:-1][::-1]
        for reduce, decoder, skip in zip(self.reduce, self.decoders, skips):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = reduce(h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        out = self.out_head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid = valid.any(dim=1, keepdim=True).expand(-1, out.shape[1], -1, -1)

        return torch.where(valid, out, torch.full_like(out, float("nan")))