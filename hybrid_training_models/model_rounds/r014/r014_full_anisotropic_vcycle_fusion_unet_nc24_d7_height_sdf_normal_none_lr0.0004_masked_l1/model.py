import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size
        self.pad = nn.ReflectionPad2d((kw // 2, kw // 2, kh // 2, kh // 2))
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kh, kw), padding=0, bias=False)

    def forward(self, x):
        return self.conv(self.pad(x))

class _AnisotropicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch
        self.in_proj = _ReflectConv(in_ch, mid, 3)
        self.h_conv = _ReflectConv(mid, mid, (1, 5))
        self.v_conv = _ReflectConv(mid, mid, (5, 1))
        self.mix = _ReflectConv(mid * 2, out_ch, 1)
        self.norm1 = nn.GroupNorm(max(1, min(8, mid)), mid)
        self.norm2 = nn.GroupNorm(max(1, min(8, out_ch)), out_ch)
        self.skip = _ReflectConv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = F.silu(self.norm1(self.in_proj(x)))
        y = torch.cat((self.h_conv(y), self.v_conv(y)), dim=1)
        y = self.norm2(self.mix(y))
        return F.silu(y + self.skip(x))

class _Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = _AnisotropicBlock(in_ch, out_ch)

    def forward(self, x):
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return self.block(x)

class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.block = _AnisotropicBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip), dim=1)
        return self.block(x)

class anisotropic_vcycle_fusion_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=7):
        super().__init__()
        depth = max(2, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = _AnisotropicBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            [_Down(channels[i - 1], channels[i]) for i in range(1, depth)]
        )

        self.bottleneck = nn.Sequential(
            _AnisotropicBlock(channels[-1], channels[-1]),
            _AnisotropicBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList(
            [
                _Up(channels[i], channels[i - 1], channels[i - 1])
                for i in range(depth - 1, 0, -1)
            ]
        )

        self.fusion = _AnisotropicBlock(channels[0] + in_channels, channels[0])
        self.head_pad = nn.ReflectionPad2d(1)
        self.head = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for down in self.encoder:
            y = down(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            y = up(y, skip)

        y = self.fusion(torch.cat((y, x_masked), dim=1))
        y = self.head(self.head_pad(y))

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if y.shape[1] == x.shape[1]:
            out_valid = valid
        else:
            out_valid = valid.all(dim=1, keepdim=True).expand(-1, y.shape[1], -1, -1)

        return torch.where(out_valid, y, torch.full_like(y, float("nan")))