import torch
import torch.nn as nn
import torch.nn.functional as F

class fno2d_lora(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, bias=True):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                fno2d_lora.ReflectConv(in_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
                fno2d_lora.ReflectConv(out_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.GELU(),
            )
            self.skip = nn.Identity() if in_ch == out_ch else fno2d_lora.ReflectConv(in_ch, out_ch, 1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class SpectralBottleneck(nn.Module):
        def __init__(self, channels, modes=32, rank=8):
            super().__init__()
            self.channels = channels
            self.modes = modes
            self.rank = min(rank, channels)
            scale = 1.0 / max(1, channels)
            self.base_real = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
            self.base_imag = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
            self.a_real = nn.Parameter(scale * torch.randn(channels, self.rank, modes, modes))
            self.a_imag = nn.Parameter(scale * torch.randn(channels, self.rank, modes, modes))
            self.b_real = nn.Parameter(scale * torch.randn(self.rank, channels, modes, modes))
            self.b_imag = nn.Parameter(scale * torch.randn(self.rank, channels, modes, modes))
            self.local = nn.Conv2d(channels, channels, 1)

        def forward(self, x):
            b, c, h, w = x.shape
            mh = min(self.modes, h)
            mw = min(self.modes, w // 2 + 1)

            x_ft = torch.fft.rfft2(x, norm="ortho")
            x_sub = x_ft[:, :, :mh, :mw]

            base = torch.complex(self.base_real[:, :, :mh, :mw], self.base_imag[:, :, :mh, :mw])
            a = torch.complex(self.a_real[:, :, :mh, :mw], self.a_imag[:, :, :mh, :mw])
            br = torch.complex(self.b_real[:, :, :mh, :mw], self.b_imag[:, :, :mh, :mw])
            lora = torch.einsum("crxy,rdxy->cdxy", a, br)
            weight = base + lora

            out_ft = torch.zeros(b, c, h, w // 2 + 1, dtype=x_ft.dtype, device=x.device)
            out_ft[:, :, :mh, :mw] = torch.einsum("bcxy,cdxy->bdxy", x_sub, weight)

            y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
            return y + self.local(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=4):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_proj = self.ReflectConv(in_channels, channels[0], 3)

        self.encoder = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.encoder.append(self.Block(prev, ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            self.Block(channels[-1], channels[-1]),
            self.SpectralBottleneck(channels[-1], modes=32, rank=8),
            nn.GELU(),
            self.Block(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        rev = list(reversed(channels))
        dec_in = channels[-1]
        for skip_ch in rev:
            self.decoder.append(self.Block(dec_in + skip_ch, skip_ch))
            dec_in = skip_ch

        self.out_proj = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3),
            nn.GELU(),
            self.ReflectConv(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        y = self.in_proj(x_masked)

        skips = []
        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i != len(self.encoder) - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for i, block in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))

        y = self.out_proj(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != y.shape[1]:
            out_valid = out_valid.any(dim=1, keepdim=True).expand_as(y)

        return torch.where(out_valid, y, torch.full_like(y, float("nan")))