import torch
import torch.nn as nn
import torch.nn.functional as F

class multigrid_spectral_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class _SpectralBlock(nn.Module):
        def __init__(self, channels, modes=24):
            super().__init__()
            self.channels = channels
            self.modes = modes
            scale = 1.0 / max(1, channels)
            self.weight_r = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
            self.weight_i = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
            self.local = nn.Conv2d(channels, channels, kernel_size=1)

        def forward(self, x):
            b, c, h, w = x.shape
            x_ft = torch.fft.rfft2(x, norm="ortho")
            mh = min(self.modes, h)
            mw = min(self.modes, x_ft.shape[-1])

            out_ft = torch.zeros(b, c, h, x_ft.shape[-1], dtype=x_ft.dtype, device=x.device)
            weight = torch.complex(self.weight_r[:, :, :mh, :mw], self.weight_i[:, :, :mh, :mw])
            out_ft[:, :, :mh, :mw] = torch.einsum("bihw,oihw->bohw", x_ft[:, :, :mh, :mw], weight)

            return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho") + self.local(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=4):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self._ConvBlock(prev, ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            self._ConvBlock(channels[-1], channels[-1]),
            self._SpectralBlock(channels[-1], modes=24),
            nn.GroupNorm(min(8, channels[-1]), channels[-1]),
            nn.GELU(),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(channels):
            self.up_projs.append(nn.Conv2d(prev, ch, kernel_size=1))
            self.decoders.append(self._ConvBlock(ch * 2, ch))
            prev = ch

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked
        for enc in self.encoders:
            y = enc(y)
            skips.append(y)
            y = self.down(y)

        y = self.bottleneck(y)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if y.shape[1] != out_valid.shape[1]:
            out_valid = out_valid.any(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)

        return torch.where(out_valid, y, torch.full_like(y, float("nan")))


