import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ReflectionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ReflectionConv2d(in_channels, out_channels, 3, bias=False)
        self.norm = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            ReflectionConv2d(out_channels, out_channels, 3, bias=False),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class TensorizedSpectralAdapter(nn.Module):
    def __init__(self, channels, modes_h=16, modes_w=16):
        super().__init__()
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.pre = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
        self.post = nn.Conv2d(channels, channels, 1, padding=0, bias=False)
        self.weight_real = nn.Parameter(torch.randn(2, channels, modes_h, modes_w) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(2, channels, modes_h, modes_w) * 0.02)
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        h, w = x.shape[-2:]
        z = self.pre(x)
        fft_input = z if z.dtype in (torch.float32, torch.float64) else z.float()
        spec = torch.fft.rfft2(fft_input, s=(h, w), norm="ortho")
        out_spec = torch.zeros_like(spec)

        mh = min(self.modes_h, max(1, spec.shape[-2] // 2))
        mw = min(self.modes_w, spec.shape[-1])

        wr = self.weight_real.to(dtype=spec.real.dtype)
        wi = self.weight_imag.to(dtype=spec.real.dtype)

        w_pos = torch.complex(wr[0, :, :mh, :mw], wi[0, :, :mh, :mw])
        out_spec[:, :, :mh, :mw] = spec[:, :, :mh, :mw] * w_pos.unsqueeze(0)

        if spec.shape[-2] > mh:
            w_neg = torch.complex(wr[1, :, :mh, :mw], wi[1, :, :mh, :mw])
            out_spec[:, :, -mh:, :mw] = spec[:, :, -mh:, :mw] * w_neg.unsqueeze(0)

        y = torch.fft.irfft2(out_spec, s=(h, w), norm="ortho").to(dtype=x.dtype)
        y = self.post(y)
        return x + self.gamma.to(dtype=x.dtype) * y


class tensorized_spectral_adapter_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if n_c < 1:
            raise ValueError("n_c must be >= 1")

        self.depth = depth
        self.out_channels = out_channels
        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]

        encoder = []
        prev_channels = in_channels
        for channels_i in channels:
            encoder.append(ResidualBlock(prev_channels, channels_i))
            prev_channels = channels_i

        self.encoder = nn.ModuleList(encoder)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1]),
            TensorizedSpectralAdapter(channels[-1]),
            ResidualBlock(channels[-1], channels[-1]),
        )

        decoder = []
        current_channels = channels[-1]
        for skip_channels in reversed(channels[:-1]):
            decoder.append(ResidualBlock(current_channels + skip_channels, skip_channels))
            current_channels = skip_channels

        self.decoder = nn.ModuleList(decoder)
        self.final = nn.Sequential(
            ReflectionConv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(_num_groups(channels[0]), channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 1, padding=0, bias=True),
        )

        if sum(p.numel() for p in self.parameters()) >= 50_000_000:
            raise ValueError("total parameter count must be under 50M; reduce n_c or depth")

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        input_size = x_masked.shape[-2:]

        skips = []
        y = x_masked

        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.down(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat((y, skip), dim=1)
            y = block(y)

        output = self.final(y)

        if output.shape[-2:] != input_size:
            output = F.interpolate(output, size=input_size, mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand(-1, output.shape[1], -1, -1)

        if valid.shape[-2:] != output.shape[-2:]:
            valid = F.interpolate(valid.float(), size=output.shape[-2:], mode="nearest").bool()

        output = output.clone()
        output[~valid] = float("nan")
        return output


