import torch
import torch.nn as nn
import torch.nn.functional as F

def _num_groups(channels):
    for g in (8, 4, 2):
        if channels % g == 0:
            return g
    return 1

class _ReflectionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class _LocalizedIntegralDifferentialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectionConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = _ReflectionConv(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)

        self.diff_pad = nn.ReflectionPad2d(1)
        diff_kernel = torch.tensor(
            [
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
                [[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]],
            ]
        )
        self.register_buffer("diff_kernel", diff_kernel, persistent=False)
        self.diff_proj = nn.Conv2d(out_channels * 3, out_channels, 1, padding=0, bias=False)

        self.int_pad = nn.ReflectionPad2d(3)
        self.int_proj = nn.Conv2d(out_channels, out_channels, 1, padding=0, bias=False)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)

        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))

        c = h.shape[1]
        kernel = self.diff_kernel.to(device=h.device, dtype=h.dtype).repeat(c, 1, 1, 1)
        diff = F.conv2d(self.diff_pad(h), kernel, groups=c)
        diff = self.diff_proj(diff)

        integ = F.avg_pool2d(self.int_pad(h), kernel_size=7, stride=1, padding=0)
        integ = self.int_proj(integ)

        return self.act(h + diff + integ + residual)

class localized_integral_differential_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")

        channels = [n_c * min(2 ** i, 8) for i in range(depth)]

        self.stem = _LocalizedIntegralDifferentialBlock(in_channels, channels[0])
        self.encoders = nn.ModuleList(
            _LocalizedIntegralDifferentialBlock(channels[i - 1], channels[i])
            for i in range(1, depth)
        )
        self.bottleneck = _LocalizedIntegralDifferentialBlock(channels[-1], channels[-1])

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up_projs.append(nn.Conv2d(channels[i], channels[i - 1], 1, padding=0, bias=False))
            self.decoders.append(
                _LocalizedIntegralDifferentialBlock(channels[i - 1] * 2, channels[i - 1])
            )

        self.head = nn.Conv2d(channels[0], out_channels, 1, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        h = self.stem(x_masked)
        skips = [h]

        for encoder in self.encoders:
            h = F.avg_pool2d(h, kernel_size=2, stride=2, padding=0)
            h = encoder(h)
            skips.append(h)

        h = self.bottleneck(h)

        for i, (up_proj, decoder) in enumerate(zip(self.up_projs, self.decoders)):
            skip = skips[-i - 2]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = decoder(torch.cat([h, skip], dim=1))

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True)
        valid = valid.expand_as(output)

        return torch.where(valid, output, torch.full_like(output, float("nan")))