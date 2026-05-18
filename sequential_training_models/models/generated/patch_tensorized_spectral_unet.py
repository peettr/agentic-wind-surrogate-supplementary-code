import torch
import torch.nn as nn
import torch.nn.functional as F

class patch_tensorized_spectral_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                patch_tensorized_spectral_unet.ReflectConv(in_ch, out_ch),
                patch_tensorized_spectral_unet.ReflectConv(out_ch, out_ch),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class SpectralBottleneck(nn.Module):
        def __init__(self, channels, modes=12):
            super().__init__()
            self.channels = channels
            self.modes = modes
            scale = 1.0 / max(1, channels)
            self.wr = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
            self.wi = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
            self.local = patch_tensorized_spectral_unet.ConvBlock(channels, channels)

        def forward(self, x):
            b, c, h, w = x.shape
            mh = min(self.modes, h)
            mw = min(self.modes, w // 2 + 1)

            xf = torch.fft.rfft2(x, norm="ortho")
            out = torch.zeros_like(xf)

            xr = xf[:, :, :mh, :mw].real
            xi = xf[:, :, :mh, :mw].imag
            wr = self.wr[:, :, :mh, :mw]
            wi = self.wi[:, :, :mh, :mw]

            yr = torch.einsum("bihw,iohw->bohw", xr, wr) - torch.einsum("bihw,iohw->bohw", xi, wi)
            yi = torch.einsum("bihw,iohw->bohw", xr, wi) + torch.einsum("bihw,iohw->bohw", xi, wr)
            out[:, :, :mh, :mw] = torch.complex(yr, yi)

            y = torch.fft.irfft2(out, s=(h, w), norm="ortho")
            return self.local(x + y)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        chs = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.enc = nn.ModuleList()
        prev = in_channels
        for ch in chs:
            self.enc.append(self.ConvBlock(prev, ch))
            prev = ch

        self.bottleneck = self.SpectralBottleneck(chs[-1], modes=12)

        self.dec = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.dec.append(self.ConvBlock(chs[i + 1] + chs[i], chs[i]))

        self.head = nn.Sequential(
            self.ReflectConv(chs[0], chs[0]),
            nn.Conv2d(chs[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for i, block in enumerate(self.enc):
            y = block(y)
            skips.append(y)
            if i != len(self.enc) - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for block, skip in zip(self.dec, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))

        y = self.head(y)
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False) if y.shape[-2:] != x.shape[-2:] else y

        out_valid = valid if valid.shape[1] == y.shape[1] else valid[:, :1].expand(-1, y.shape[1], -1, -1)
        y = torch.where(out_valid, y, torch.full_like(y, float("nan")))
        return y