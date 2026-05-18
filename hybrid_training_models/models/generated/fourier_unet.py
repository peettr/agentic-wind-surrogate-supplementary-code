import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.net(x)

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv(in_channels, out_channels),
            _ReflectConv(out_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)

class _FourierBlock(nn.Module):
    def __init__(self, channels, modes=24):
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1.0 / max(1, channels)
        self.weight_real = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, c, h, w // 2 + 1, dtype=x_ft.dtype, device=x.device)

        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)
        weight = torch.complex(self.weight_real[:, :, :mh, :mw], self.weight_imag[:, :, :mh, :mw])

        out_ft[:, :, :mh, :mw] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :mh, :mw], weight)
        x_spec = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return self.mix(x + x_spec)

class fourier_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=4):
        super().__init__()
        depth = max(1, int(depth))
        channels = [n_c * min(2 ** i, 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(_ConvBlock(prev_channels, ch))
            prev_channels = ch

        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            _ConvBlock(bottleneck_channels, bottleneck_channels),
            _FourierBlock(bottleneck_channels),
            _ConvBlock(bottleneck_channels, bottleneck_channels)
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for skip_channels in reversed(channels[:-1]):
            self.upconvs.append(nn.ConvTranspose2d(prev_channels, skip_channels, kernel_size=2, stride=2))
            self.decoders.append(_ConvBlock(skip_channels * 2, skip_channels))
            prev_channels = skip_channels

        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=0)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            if i < len(self.encoders) - 1:
                skips.append(y)
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            y = upconv(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = decoder(y)

        y = self.out(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != y.shape[1]:
            valid_out = valid_out.all(dim=1, keepdim=True).expand_as(y)
        y = torch.where(valid_out, y, torch.full_like(y, float("nan")))
        return y