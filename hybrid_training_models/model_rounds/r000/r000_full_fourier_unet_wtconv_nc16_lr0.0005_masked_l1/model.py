import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class _RefDepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=0,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(self.pad(x)))


class _WTConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pad3 = nn.ReflectionPad2d(1)
        self.pad5 = nn.ReflectionPad2d(2)
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=0, groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, kernel_size=5, padding=0, groups=channels, bias=False)
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        y = self.dw3(self.pad3(x))
        if x.shape[-2] > 2 and x.shape[-1] > 2:
            y = y + self.dw5(self.pad5(x))
        return self.mix(y)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _RefDepthwiseSeparableConv(in_channels, out_channels)
        self.norm1 = _norm(out_channels)
        self.conv2 = _RefDepthwiseSeparableConv(out_channels, out_channels)
        self.norm2 = _norm(out_channels)
        self.wtconv = _WTConv(out_channels)
        self.norm3 = _norm(out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        x = self.norm3(self.wtconv(x))
        return self.act(x + residual)


class _FourierBlock(nn.Module):
    def __init__(self, channels, modes=16):
        super().__init__()
        self.modes_y = modes
        self.modes_x = modes
        self.weight_real = nn.Parameter(torch.randn(1, channels, modes, modes) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(1, channels, modes, modes) * 0.02)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=False)
        self.norm = _norm(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        h, w = x.shape[-2:]
        my = min(self.modes_y, h)
        mx = min(self.modes_x, w // 2 + 1)

        x_fft = torch.fft.rfft2(x, norm="ortho")
        y_fft = torch.zeros_like(x_fft)

        wr = self.weight_real[..., :my, :mx].to(dtype=x_fft.real.dtype, device=x_fft.device)
        wi = self.weight_imag[..., :my, :mx].to(dtype=x_fft.real.dtype, device=x_fft.device)
        weight = torch.complex(wr, wi)

        y_fft[:, :, :my, :mx] = x_fft[:, :, :my, :mx] * weight

        if h > my:
            neg_my = min(my, h - my)
            y_fft[:, :, -neg_my:, :mx] = x_fft[:, :, -neg_my:, :mx] * weight[..., :neg_my, :]

        y = torch.fft.irfft2(y_fft, s=(h, w), norm="ortho")
        y = self.proj(y)
        return self.act(self.norm(residual + y))


class fourier_unet_wtconv(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=4):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        bottleneck_channels = min(n_c * (2 ** depth), n_c * 8)

        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoder.append(_ConvBlock(prev_channels, ch))
            prev_channels = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            _ConvBlock(channels[-1], bottleneck_channels),
            _FourierBlock(bottleneck_channels, modes=16),
        )

        self.decoder = nn.ModuleList()
        prev_channels = bottleneck_channels
        for skip_channels in reversed(channels):
            self.decoder.append(_ConvBlock(prev_channels + skip_channels, skip_channels))
            prev_channels = skip_channels

        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x).any(dim=1, keepdim=True)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked

        for block in self.encoder:
            y = block(y)
            skips.append(y)
            y = self.down(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        output = self.out_conv(y)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid = valid.expand(-1, output.shape[1], -1, -1)
        output = output.clone()
        output[~valid] = float("nan")
        return output