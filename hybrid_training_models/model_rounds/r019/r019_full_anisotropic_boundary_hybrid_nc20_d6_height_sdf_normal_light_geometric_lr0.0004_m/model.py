import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _safe_reflect_pad(x, pad_lr, pad_tb):
    h, w = x.shape[-2:]
    use_l = min(pad_lr, max(0, w - 1))
    use_t = min(pad_tb, max(0, h - 1))
    if use_l > 0 or use_t > 0:
        x = F.pad(x, (use_l, use_l, use_t, use_t), mode="reflect")
    rem_l = pad_lr - use_l
    rem_t = pad_tb - use_t
    if rem_l > 0 or rem_t > 0:
        x = F.pad(x, (rem_l, rem_l, rem_t, rem_t), mode="replicate")
    return x


class anisotropic_boundary_hybrid(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, groups=1, bias=False):
            super().__init__()
            if isinstance(kernel_size, tuple):
                self.pad_tb = kernel_size[0] // 2
                self.pad_lr = kernel_size[1] // 2
            else:
                self.pad_tb = kernel_size // 2
                self.pad_lr = kernel_size // 2
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, groups=groups, bias=bias)

        def forward(self, x):
            x = _safe_reflect_pad(x, self.pad_lr, self.pad_tb)
            return self.conv(x)

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.proj = nn.Identity() if in_ch == out_ch else anisotropic_boundary_hybrid.ReflectConv(in_ch, out_ch, 1, bias=False)
            self.net = nn.Sequential(
                anisotropic_boundary_hybrid.ReflectConv(in_ch, out_ch, 3, bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                anisotropic_boundary_hybrid.ReflectConv(out_ch, out_ch, (1, 5), bias=False),
                _gn(out_ch),
                nn.SiLU(inplace=True),
                anisotropic_boundary_hybrid.ReflectConv(out_ch, out_ch, (5, 1), bias=False),
                _gn(out_ch),
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.proj(x))

    class Bottleneck(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.local = anisotropic_boundary_hybrid.Block(ch, ch)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, max(1, ch // 4), 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(1, ch // 4), ch, 1),
                nn.Sigmoid(),
            )
            self.mix = anisotropic_boundary_hybrid.ReflectConv(ch, ch, 1, bias=False)

        def forward(self, x):
            y = self.local(x)
            return self.mix(y * self.gate(y) + x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = in_channels

        for ch in channels:
            self.encoders.append(self.Block(prev, ch))
            self.downs.append(nn.AvgPool2d(2))
            prev = ch

        self.bottleneck = self.Bottleneck(channels[-1])

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for ch in reversed(channels):
            self.up_projs.append(self.ReflectConv(prev, ch, 1, bias=False))
            self.decoders.append(self.Block(ch * 2, ch))
            prev = ch

        self.head = nn.Sequential(
            self.ReflectConv(prev, prev, 3, bias=False),
            _gn(prev),
            nn.SiLU(inplace=True),
            self.ReflectConv(prev, out_channels, 1, bias=True),
        )

    def forward(self, x):
        nan_mask = torch.isnan(x)
        valid = ~nan_mask
        x_masked = torch.where(nan_mask, torch.zeros_like(x), x)

        skips = []
        y = x_masked

        for enc, down in zip(self.encoders, self.downs):
            y = enc(y)
            skips.append(y)
            if y.shape[-2] >= 2 and y.shape[-1] >= 2:
                y = down(y)

        y = self.bottleneck(y)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips)):
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(y)

        return torch.where(valid, y, torch.full_like(y, float("nan")))


