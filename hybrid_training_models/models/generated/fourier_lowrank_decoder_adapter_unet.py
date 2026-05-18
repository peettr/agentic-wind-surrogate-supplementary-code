import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    groups = min(8, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
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
            ReflectConv2d(in_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.GELU(),
            ReflectConv2d(out_channels, out_channels, 3, bias=False),
            _gn(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class FourierLowRankAdapter(nn.Module):
    def __init__(self, channels, modes=24, rank=8):
        super().__init__()
        self.modes = modes
        self.rank = min(rank, channels)
        scale = 1.0 / max(1, channels)
        self.left_real = nn.Parameter(scale * torch.randn(channels, self.rank))
        self.left_imag = nn.Parameter(scale * torch.randn(channels, self.rank))
        self.right_real = nn.Parameter(scale * torch.randn(self.rank, channels))
        self.right_imag = nn.Parameter(scale * torch.randn(self.rank, channels))

    def forward(self, x):
        b, c, h, w = x.shape
        mh = min(self.modes, h)
        mw = min(self.modes, w // 2 + 1)

        x_ft = torch.fft.rfft2(x, norm="ortho")
        low = x_ft[:, :, :mh, :mw]

        left = torch.complex(self.left_real, self.left_imag)
        right = torch.complex(self.right_real, self.right_imag)
        weight = left @ right

        low = torch.einsum("bchw,co->bohw", low, weight)

        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :mh, :mw] = low
        out = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return x + out


class fourier_lowrank_decoder_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.depth = depth
        self.channels = [n_c * min(2 ** i, 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in self.channels:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            ConvBlock(self.channels[-1], self.channels[-1]),
            FourierLowRankAdapter(self.channels[-1], modes=20, rank=8),
            ConvBlock(self.channels[-1], self.channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.adapters = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            in_ch = self.channels[i + 1]
            skip_ch = self.channels[i]
            self.up_projs.append(ReflectConv2d(in_ch, skip_ch, 1, bias=False))
            self.adapters.append(FourierLowRankAdapter(skip_ch, modes=20, rank=8))
            self.decoders.append(ConvBlock(skip_ch * 2, skip_ch))

        self.head = ReflectConv2d(self.channels[0], out_channels, 1, bias=True)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for i, decoder in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = self.up_projs[i](h)
            h = self.adapters[i](h)
            h = decoder(torch.cat([h, skip], dim=1))

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(output)

        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output