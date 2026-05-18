import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, groups=1, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            _ReflectConv(in_ch, out_ch, 3),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
            _ReflectConv(out_ch, out_ch, 3),
            nn.GroupNorm(min(8, out_ch), out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


class _SSMBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw_h = nn.Conv2d(ch, ch, (1, 7), padding=0, groups=ch)
        self.dw_v = nn.Conv2d(ch, ch, (7, 1), padding=0, groups=ch)
        self.pad_h = nn.ReflectionPad2d((3, 3, 0, 0))
        self.pad_v = nn.ReflectionPad2d((0, 0, 3, 3))
        self.mix = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, max(1, ch // 4), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, ch // 4), ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.dw_h(self.pad_h(x)) + self.dw_v(self.pad_v(x))
        y = self.mix(y)
        y = y * self.gate(y)
        return x + self.norm(y)


class _AttentionGate(nn.Module):
    def __init__(self, skip_ch, gate_ch):
        super().__init__()
        mid = max(1, min(skip_ch, gate_ch) // 2)
        self.skip_proj = nn.Conv2d(skip_ch, mid, 1)
        self.gate_proj = nn.Conv2d(gate_ch, mid, 1)
        self.psi = nn.Conv2d(mid, 1, 1)

    def forward(self, skip, gate):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        a = torch.sigmoid(self.psi(F.silu(self.skip_proj(skip) + self.gate_proj(gate))))
        return skip * a


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = _ConvBlock(in_ch, out_ch)
        self.ssm = _SSMBlock(out_ch)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        x = self.block(x)
        return self.ssm(x)


class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.attn = _AttentionGate(skip_ch, in_ch)
        self.block = _ConvBlock(in_ch + skip_ch, out_ch)
        self.ssm = _SSMBlock(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attn(skip, x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return self.ssm(x)


class attn_ssm_skip_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        self.out_channels = out_channels
        self.depth = max(1, depth)
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(self.depth)]

        self.stem = _ConvBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            [_Down(channels[i - 1], channels[i]) for i in range(1, self.depth)]
        )

        bottleneck_ch = channels[-1]
        self.bottleneck = nn.Sequential(
            _ConvBlock(bottleneck_ch, bottleneck_ch),
            _SSMBlock(bottleneck_ch),
            _SSMBlock(bottleneck_ch),
        )

        self.decoder = nn.ModuleList()
        for i in range(self.depth - 2, -1, -1):
            self.decoder.append(_Up(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for down in self.encoder:
            y = down(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            y = up(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.expand(-1, y.shape[1], -1, -1)

        return torch.where(valid, y, torch.full_like(y, float("nan")))