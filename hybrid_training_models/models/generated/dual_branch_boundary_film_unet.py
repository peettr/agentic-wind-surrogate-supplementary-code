import torch
import torch.nn as nn
import torch.nn.functional as F

class dual_branch_boundary_film_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, groups=groups, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                dual_branch_boundary_film_unet.ReflectConv(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                dual_branch_boundary_film_unet.ReflectConv(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class BoundaryFiLM(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.to_gamma_beta = nn.Sequential(
                dual_branch_boundary_film_unet.ReflectConv(1, channels, 3, bias=True),
                nn.SiLU(inplace=True),
                dual_branch_boundary_film_unet.ReflectConv(channels, channels * 2, 3, bias=True),
            )

        def forward(self, feat, boundary):
            gb = self.to_gamma_beta(boundary)
            gamma, beta = gb.chunk(2, dim=1)
            return feat * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        chans = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = chans

        self.boundary_stem = nn.Sequential(
            self.ReflectConv(1, n_c, 3, bias=False),
            nn.GroupNorm(min(8, n_c), n_c),
            nn.SiLU(inplace=True),
            self.ReflectConv(n_c, 1, 3, bias=True),
        )

        self.encoders = nn.ModuleList()
        self.films = nn.ModuleList()
        prev = in_channels
        for ch in chans:
            self.encoders.append(self.ConvBlock(prev, ch))
            self.films.append(self.BoundaryFiLM(ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.ConvBlock(chans[-1], chans[-1])

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.ConvTranspose2d(chans[i + 1], chans[i], kernel_size=2, stride=2))
            self.decoders.append(self.ConvBlock(chans[i] * 2, chans[i]))

        self.head = nn.Sequential(
            self.ReflectConv(chans[0], chans[0], 3, bias=False),
            nn.GroupNorm(min(8, chans[0]), chans[0]),
            nn.SiLU(inplace=True),
            self.ReflectConv(chans[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        boundary = (~valid).to(dtype=x_masked.dtype)
        boundary = self.boundary_stem(boundary)

        skips = []
        h = x_masked
        b = boundary

        for i, enc in enumerate(self.encoders):
            h = enc(h)
            h = self.films[i](h, b)
            skips.append(h)
            if i < self.depth - 1:
                h = self.down(h)
                b = self.down(b)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out.shape[1] != out_valid.shape[1]:
            out_valid = out_valid.expand(-1, out.shape[1], -1, -1)
        out = out.masked_fill(~out_valid, float("nan"))
        return out


