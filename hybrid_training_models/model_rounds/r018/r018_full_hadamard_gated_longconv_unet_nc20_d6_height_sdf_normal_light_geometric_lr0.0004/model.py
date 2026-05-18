import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _reflect_pad2d(x, pad_h, pad_w):
    if pad_h == 0 and pad_w == 0:
        return x
    H, W = x.shape[-2:]
    pad_h = min(pad_h, max(0, H - 1))
    pad_w = min(pad_w, max(0, W - 1))
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")


def _dw_longconv(x, weight, kernel_size):
    H, W = x.shape[-2:]
    full_pad = kernel_size // 2
    pad_h = min(full_pad, max(0, H - 1))
    pad_w = min(full_pad, max(0, W - 1))
    kh = 2 * pad_h + 1
    kw = 2 * pad_w + 1
    if kh != kernel_size or kw != kernel_size:
        off_h = (kernel_size - kh) // 2
        off_w = (kernel_size - kw) // 2
        w = weight[..., off_h:off_h + kh, off_w:off_w + kw].contiguous()
    else:
        w = weight
    x_p = _reflect_pad2d(x, pad_h, pad_w)
    groups = weight.shape[0]
    return F.conv2d(x_p, w, bias=None, stride=1, padding=0, groups=groups)


class hadamard_gated_longconv_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, bias=False)
            self.norm1 = _gn(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, bias=False)
            self.norm2 = _gn(out_ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            x = _reflect_pad2d(x, 1, 1)
            x = self.act(self.norm1(self.conv1(x)))
            x = _reflect_pad2d(x, 1, 1)
            x = self.act(self.norm2(self.conv2(x)))
            return x

    class GatedLongConv(nn.Module):
        def __init__(self, channels, kernel_size=15):
            super().__init__()
            self.channels = channels
            self.kernel_size = int(kernel_size)
            if self.kernel_size % 2 == 0:
                self.kernel_size += 1
            self.norm = _gn(channels)
            self.v_dw = nn.Parameter(torch.empty(channels, 1, self.kernel_size, self.kernel_size))
            self.g_dw = nn.Parameter(torch.empty(channels, 1, self.kernel_size, self.kernel_size))
            nn.init.kaiming_uniform_(self.v_dw, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.g_dw, a=math.sqrt(5))
            self.v_pw = nn.Conv2d(channels, channels, 1, bias=False)
            self.g_pw = nn.Conv2d(channels, channels, 1, bias=True)
            self.proj = nn.Conv2d(channels, channels, 1, bias=False)

        def forward(self, x):
            y = self.norm(x)
            v = self.v_pw(_dw_longconv(y, self.v_dw, self.kernel_size))
            g = self.g_pw(_dw_longconv(y, self.g_dw, self.kernel_size))
            y = v * torch.sigmoid(g)
            return x + self.proj(y)

    class UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.block = hadamard_gated_longconv_unet.ConvBlock(out_ch + skip_ch, out_ch)
            self.longconv = hadamard_gated_longconv_unet.GatedLongConv(out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            x = self.block(x)
            return self.longconv(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.ConvBlock(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        self.encoder_longconvs = nn.ModuleList()
        for i in range(1, depth):
            self.encoders.append(self.ConvBlock(channels[i - 1], channels[i]))
            self.encoder_longconvs.append(self.GatedLongConv(channels[i]))

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.GatedLongConv(channels[-1], kernel_size=21),
            self.GatedLongConv(channels[-1], kernel_size=21),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self.UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head_conv = nn.Conv2d(channels[0], channels[0], 3, bias=False)
        self.head_norm = _gn(channels[0])
        self.head_act = nn.SiLU(inplace=True)
        self.head_out = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for enc, longconv in zip(self.encoders, self.encoder_longconvs):
            h = F.avg_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)
            h = enc(h)
            h = longconv(h)
            skips.append(h)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            h = dec(h, skip)

        h = _reflect_pad2d(h, 1, 1)
        h = self.head_act(self.head_norm(self.head_conv(h)))
        out = self.head_out(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            out_valid = out_valid[:, :1].expand(-1, out.shape[1], -1, -1)

        return torch.where(out_valid, out, torch.full_like(out, float("nan")))


