import torch
import torch.nn as nn
import torch.nn.functional as F

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
            ReflectionConv2d(in_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class SpectralMixer(nn.Module):
    def __init__(self, channels, modes=24):
        super().__init__()
        self.channels = channels
        self.modes = modes
        scale = 1.0 / max(1, channels)
        self.weight_real = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(channels, channels, modes, modes))
        self.proj = nn.Sequential(
            ReflectionConv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        m_h = min(self.modes, h)
        m_w = min(self.modes, w // 2 + 1)

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(b, c, h, w // 2 + 1, dtype=x_ft.dtype, device=x.device)

        w_complex = torch.complex(
            self.weight_real[:, :, :m_h, :m_w],
            self.weight_imag[:, :, :m_h, :m_w],
        )
        out_ft[:, :, :m_h, :m_w] = torch.einsum(
            "bihw,oihw->bohw",
            x_ft[:, :, :m_h, :m_w],
            w_complex,
        )

        y = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return x + self.proj(y)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class tiled_spectral_mixer_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=5):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.channels = channels

        self.stem = ConvBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(DownBlock(channels[i - 1], channels[i]))

        bottleneck_channels = channels[-1]
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_channels, bottleneck_channels),
            SpectralMixer(bottleneck_channels, modes=24),
            ConvBlock(bottleneck_channels, bottleneck_channels),
        )

        self.decoder = nn.ModuleList()
        current_channels = bottleneck_channels
        for i in range(depth - 2, -1, -1):
            self.decoder.append(UpBlock(current_channels, channels[i], channels[i]))
            current_channels = channels[i]

        self.head = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            ReflectionConv2d(channels[0], out_channels, 1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for block in self.encoder:
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = block(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        nan_mask = ~valid
        if nan_mask.shape[1] != y.shape[1]:
            nan_mask = nan_mask[:, :1].expand(-1, y.shape[1], -1, -1)

        output = y.clone()
        output[nan_mask] = float("nan")
        return output