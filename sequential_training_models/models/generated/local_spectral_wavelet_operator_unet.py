import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.pad(x))

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv2d(in_channels, out_channels, 3),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            _ReflectConv2d(out_channels, out_channels, 3),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.net(x) + self.skip(x)

class _LocalSpectralWaveletBlock(nn.Module):
    def __init__(self, channels, modes=16):
        super().__init__()
        self.channels = channels
        self.modes = modes

        self.local = nn.Sequential(
            _ReflectConv2d(channels, channels, 3, groups=channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
        )

        self.low = nn.Sequential(
            nn.AvgPool2d(2, 2),
            _ReflectConv2d(channels, channels, 3),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
        )

        self.high = nn.Sequential(
            _ReflectConv2d(channels, channels, 3, groups=channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
        )

        scale = channels ** -0.5
        self.spectral_weight = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))
        self.mix = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.SiLU(inplace=True)

    def _spectral(self, x):
        b, c, h, w = x.shape
        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, c, h, w // 2 + 1, dtype=x_ft.dtype, device=x.device)

        weight = torch.view_as_complex(self.spectral_weight[:, :, :mh, :mw, :].contiguous())
        out_ft[:, :, :mh, :mw] = torch.einsum("bihw,iohw->bohw", x_ft[:, :, :mh, :mw], weight)

        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")

    def forward(self, x):
        local = self.local(x)

        low = self.low(x)
        low = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)

        blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = self.high(x - blurred)

        spectral = self._spectral(x)
        y = self.mix(torch.cat([local + spectral, low, high], dim=1))
        return self.act(self.norm(y) + x)

class local_spectral_wavelet_operator_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        depth = max(1, int(depth))
        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        self.out_channels = out_channels
        self.encoders = nn.ModuleList()
        self.operators = nn.ModuleList()
        self.downs = nn.ModuleList()

        prev = in_channels
        for ch in channels:
            self.encoders.append(_ConvBlock(prev, ch))
            self.operators.append(_LocalSpectralWaveletBlock(ch, modes=16))
            self.downs.append(nn.AvgPool2d(2, 2))
            prev = ch

        self.bottleneck = nn.Sequential(
            _ConvBlock(channels[-1], channels[-1]),
            _LocalSpectralWaveletBlock(channels[-1], modes=16),
            _ConvBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        self.up_projs = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up_projs.append(nn.Conv2d(channels[i], channels[i - 1], 1, bias=False))
            self.decoders.append(_ConvBlock(channels[i - 1] * 2, channels[i - 1]))

        self.head = nn.Sequential(
            _ReflectConv2d(channels[0], channels[0], 3),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = self.operators[i](encoder(h))
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.downs[i](h)

        h = self.bottleneck(h)

        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = self.up_projs[i](h)
            h = decoder(torch.cat([h, skip], dim=1))

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand(-1, output.shape[1], -1, -1)

        output = output.clone()
        output[~valid] = float("nan")
        return output