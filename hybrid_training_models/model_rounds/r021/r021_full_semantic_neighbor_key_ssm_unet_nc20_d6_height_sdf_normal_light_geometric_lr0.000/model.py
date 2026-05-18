import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = min(8, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class semantic_neighbor_key_ssm_unet(nn.Module):
    class RefConv2d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
            self.conv1 = semantic_neighbor_key_ssm_unet.RefConv2d(in_ch, out_ch, 3, bias=False)
            self.norm1 = _gn(out_ch)
            self.conv2 = semantic_neighbor_key_ssm_unet.RefConv2d(out_ch, out_ch, 3, bias=False)
            self.norm2 = _gn(out_ch)

        def forward(self, x):
            r = self.proj(x)
            x = F.silu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            return F.silu(x + r)

    class SemanticNeighborKeySSM(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.key = nn.Conv2d(ch, ch, 1, bias=False)
            self.value = nn.Conv2d(ch, ch, 1, bias=False)
            self.gate = nn.Conv2d(ch, ch, 1, bias=True)
            self.dw_h = nn.Conv2d(ch, ch, (1, 3), padding=0, groups=ch, bias=False)
            self.dw_v = nn.Conv2d(ch, ch, (3, 1), padding=0, groups=ch, bias=False)
            self.pad_h = nn.ReflectionPad2d((1, 1, 0, 0))
            self.pad_v = nn.ReflectionPad2d((0, 0, 1, 1))
            self.out = nn.Conv2d(ch, ch, 1, bias=False)

        def forward(self, x):
            k = torch.sigmoid(self.key(x))
            val = self.value(x)
            h = self.dw_h(self.pad_h(val * k))
            v = self.dw_v(self.pad_v(val * k))
            g = torch.sigmoid(self.gate(x))
            return x + self.out((h + v) * g)

    class Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = semantic_neighbor_key_ssm_unet.Block(in_ch, out_ch)
            self.ssm = semantic_neighbor_key_ssm_unet.SemanticNeighborKeySSM(out_ch)

        def forward(self, x):
            x = self.block(x)
            x = self.ssm(x)
            return x

    class Up(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.block = semantic_neighbor_key_ssm_unet.Block(in_ch + skip_ch, out_ch)
            self.ssm = semantic_neighbor_key_ssm_unet.SemanticNeighborKeySSM(out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.block(x)
            x = self.ssm(x)
            return x

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.enc = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.enc.append(self.Down(prev, ch))
            prev = ch

        self.bottleneck = nn.Sequential(
            self.Block(channels[-1], channels[-1]),
            self.SemanticNeighborKeySSM(channels[-1]),
            self.Block(channels[-1], channels[-1]),
        )

        self.dec = nn.ModuleList()
        rev = list(reversed(channels))
        prev = rev[0]
        for skip_ch in rev[1:]:
            self.dec.append(self.Up(prev, skip_ch, skip_ch))
            prev = skip_ch

        self.head = nn.Sequential(
            self.Block(channels[0], channels[0]),
            self.RefConv2d(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, enc in enumerate(self.enc):
            h = enc(h)
            skips.append(h)
            if i != len(self.enc) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        skips = skips[:-1][::-1]
        for up, skip in zip(self.dec, skips):
            h = up(h, skip)

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            out_valid = out_valid[:, :1].expand(-1, out.shape[1], -1, -1)
        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out


