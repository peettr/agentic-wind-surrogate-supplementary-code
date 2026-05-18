import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = min(8, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class tensorized_fourier_local_global_adapter_unet(nn.Module):
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
                tensorized_fourier_local_global_adapter_unet.ReflectionConv2d(in_channels, out_channels, 3, bias=False),
                _gn(out_channels),
                nn.GELU(),
                tensorized_fourier_local_global_adapter_unet.ReflectionConv2d(out_channels, out_channels, 3, bias=False),
                _gn(out_channels),
                nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class FourierAdapter(nn.Module):
        def __init__(self, channels, modes=16):
            super().__init__()
            self.channels = channels
            self.modes = modes
            self.local = nn.Sequential(
                tensorized_fourier_local_global_adapter_unet.ReflectionConv2d(channels, channels, 3, bias=False),
                _gn(channels),
                nn.GELU(),
            )
            scale = 1.0 / max(1, channels)
            self.weight_real_h = nn.Parameter(scale * torch.randn(channels, modes))
            self.weight_imag_h = nn.Parameter(scale * torch.randn(channels, modes))
            self.weight_real_w = nn.Parameter(scale * torch.randn(channels, modes))
            self.weight_imag_w = nn.Parameter(scale * torch.randn(channels, modes))
            self.proj = tensorized_fourier_local_global_adapter_unet.ReflectionConv2d(channels, channels, 1, bias=True)
            self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x):
            b, c, h, w = x.shape
            mh = min(self.modes, h)
            mw = min(self.modes, (w // 2) + 1)

            x_ft = torch.fft.rfft2(x, norm="ortho")
            out_ft = torch.zeros_like(x_ft)

            wr = self.weight_real_h[:, :mh].unsqueeze(-1) * self.weight_real_w[:, :mw].unsqueeze(-2)
            wi = self.weight_imag_h[:, :mh].unsqueeze(-1) * self.weight_imag_w[:, :mw].unsqueeze(-2)
            weight = torch.complex(wr, wi).unsqueeze(0)

            out_ft[:, :, :mh, :mw] = x_ft[:, :, :mh, :mw] * weight
            global_x = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
            global_x = self.proj(global_x)

            return self.local(x) + torch.tanh(self.gate) * global_x

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev, ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.FourierAdapter(channels[-1], modes=16),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upconvs.append(self.ReflectionConv2d(channels[i + 1], channels[i], 1, bias=True))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.GELU(),
            self.ReflectionConv2d(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i < self.depth - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(y)
            y = torch.cat([y, skip], dim=1)
            y = decoder(y)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        output_valid = valid
        if output_valid.shape[1] != y.shape[1]:
            output_valid = output_valid[:, :1].expand(-1, y.shape[1], -1, -1)

        y = torch.where(output_valid, y, torch.full_like(y, float("nan")))
        return y


