import torch
import torch.nn as nn
import torch.nn.functional as F

class RefConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, groups=groups)

    def forward(self, x):
        return self.conv(self.pad(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.net = nn.Sequential(
            RefConv2d(in_channels, out_channels, 3),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            RefConv2d(out_channels, out_channels, 3),
            nn.GroupNorm(min(8, out_channels), out_channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + self.proj(x))

class KANLikeBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.a = nn.Conv2d(channels, hidden, 1)
        self.b = nn.Conv2d(channels, hidden, 1)
        self.c = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        y = F.gelu(self.a(x)) * torch.tanh(self.b(x))
        return x + self.c(y)

class TruncatedSpectralBlock(nn.Module):
    def __init__(self, channels, modes=20):
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1.0 / max(1, channels)
        self.wr = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        self.wi = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        self.local = nn.Conv2d(channels, channels, 1)

    def compl_mul2d(self, x, wr, wi):
        xr = x.real
        xi = x.imag
        real = torch.einsum("bixy,ioxy->boxy", xr, wr) - torch.einsum("bixy,ioxy->boxy", xi, wi)
        imag = torch.einsum("bixy,ioxy->boxy", xr, wi) + torch.einsum("bixy,ioxy->boxy", xi, wr)
        return torch.complex(real, imag)

    def forward(self, x):
        b, c, h, w = x.shape
        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, c, h, w // 2 + 1, dtype=x_ft.dtype, device=x.device)
        out_ft[:, :, :mh, :mw] = self.compl_mul2d(
            x_ft[:, :, :mh, :mw],
            self.wr[:, :, :mh, :mw],
            self.wi[:, :, :mh, :mw],
        )
        y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return F.gelu(y + self.local(x))

class kan_fno_hybrid(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=5):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(ConvBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], channels[-1]),
            TruncatedSpectralBlock(channels[-1], modes=20),
            KANLikeBlock(channels[-1]),
            ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(ConvBlock(channels[i + 1] + channels[i], channels[i]))

        self.head = nn.Sequential(
            RefConv2d(channels[0], channels[0], 3),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for block in self.encoder:
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        y = torch.where(valid, y, torch.full_like(y, float("nan")))
        return y