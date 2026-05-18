import torch
import torch.nn as nn
import torch.nn.functional as F


class _ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            _ReflectConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            _ReflectConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _GeometryGate(nn.Module):
    def __init__(self, geom_channels, feat_channels):
        super().__init__()
        hidden = max(8, min(feat_channels, geom_channels * 2))
        self.net = nn.Sequential(
            _ReflectConv2d(geom_channels, hidden, 3, bias=True),
            nn.SiLU(inplace=True),
            _ReflectConv2d(hidden, feat_channels * 2, 3, bias=True),
        )

    def forward(self, feat, geom):
        gamma, beta = self.net(geom).chunk(2, dim=1)
        return feat * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * beta


class _AFNO2d(nn.Module):
    def __init__(self, channels, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=0.5):
        super().__init__()
        self.channels = channels
        self.num_blocks = max(1, min(num_blocks, channels))
        while channels % self.num_blocks != 0:
            self.num_blocks -= 1
        self.block_size = channels // self.num_blocks
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction

        scale = 0.02
        self.w1 = nn.Parameter(scale * torch.randn(self.num_blocks, self.block_size, self.block_size * 2))
        self.b1 = nn.Parameter(scale * torch.randn(self.num_blocks, self.block_size * 2))
        self.w2 = nn.Parameter(scale * torch.randn(self.num_blocks, self.block_size, self.block_size * 2))
        self.b2 = nn.Parameter(scale * torch.randn(self.num_blocks, self.block_size * 2))

    def forward(self, x):
        residual = x
        b, c, h, w = x.shape

        x_ft = torch.fft.rfft2(x.float(), norm="ortho")
        h_modes = max(1, int(h * self.hard_thresholding_fraction))
        w_modes = max(1, int((w // 2 + 1) * self.hard_thresholding_fraction))
        h_start = (h - h_modes) // 2

        y = torch.zeros_like(x_ft)
        x_crop = x_ft[:, :, h_start:h_start + h_modes, :w_modes]
        x_crop = x_crop.permute(0, 2, 3, 1).reshape(b, h_modes, w_modes, self.num_blocks, self.block_size)

        xr = x_crop.real
        xi = x_crop.imag

        o1r = F.relu(
            torch.einsum("bhwki,kio->bhwko", xr, self.w1[:, :, :self.block_size])
            - torch.einsum("bhwki,kio->bhwko", xi, self.w1[:, :, self.block_size:])
            + self.b1[:, :self.block_size]
        )
        o1i = F.relu(
            torch.einsum("bhwki,kio->bhwko", xi, self.w1[:, :, :self.block_size])
            + torch.einsum("bhwki,kio->bhwko", xr, self.w1[:, :, self.block_size:])
            + self.b1[:, self.block_size:]
        )

        o2r = (
            torch.einsum("bhwki,kio->bhwko", o1r, self.w2[:, :, :self.block_size])
            - torch.einsum("bhwki,kio->bhwko", o1i, self.w2[:, :, self.block_size:])
            + self.b2[:, :self.block_size]
        )
        o2i = (
            torch.einsum("bhwki,kio->bhwko", o1i, self.w2[:, :, :self.block_size])
            + torch.einsum("bhwki,kio->bhwko", o1r, self.w2[:, :, self.block_size:])
            + self.b2[:, self.block_size:]
        )

        out_crop = torch.stack((o2r, o2i), dim=-1)
        out_crop = F.softshrink(out_crop, lambd=self.sparsity_threshold)
        out_crop = torch.view_as_complex(out_crop.contiguous())
        out_crop = out_crop.reshape(b, h_modes, w_modes, c).permute(0, 3, 1, 2)
        y[:, :, h_start:h_start + h_modes, :w_modes] = out_crop

        y = torch.fft.irfft2(y, s=(h, w), norm="ortho")
        return residual + y.to(dtype=residual.dtype)


class geometry_modulated_afno_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoder = nn.ModuleList()
        self.geom_gates = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoder.append(_ConvBlock(prev, ch))
            self.geom_gates.append(_GeometryGate(in_channels, ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Sequential(
            _AFNO2d(channels[-1]),
            _ConvBlock(channels[-1], channels[-1]),
            _AFNO2d(channels[-1]),
        )

        self.up_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_blocks.append(_ConvBlock(channels[i + 1] + channels[i], channels[i]))

        self.head = nn.Sequential(
            _ReflectConv2d(channels[0], channels[0], 3, bias=False),
            nn.SiLU(inplace=True),
            _ReflectConv2d(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))
        geom = x_masked

        skips = []
        h = x_masked
        for i, block in enumerate(self.encoder):
            h = block(h)
            geom_i = F.interpolate(geom, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = self.geom_gates[i](h, geom_i)
            skips.append(h)
            if i < self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels == self.in_channels:
            out_valid = valid
        else:
            out_valid = valid.any(dim=1, keepdim=True).expand(-1, self.out_channels, -1, -1)
        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out


