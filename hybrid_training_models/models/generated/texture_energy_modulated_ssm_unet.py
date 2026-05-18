import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch):
    groups = min(8, ch)
    while ch % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, ch)


class _ReflectConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, groups=1):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, groups=groups, bias=False)
        self.norm = _gn(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(self.pad(x))))


class _TextureEnergyGate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.local = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, padding=0, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
        dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
        e = torch.sqrt(dx * dx + dy * dy + 1e-6)
        return x * (1.0 + self.local(e))


class _SSM2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.in_proj = nn.Conv2d(ch, ch, 1, bias=True)
        self.dw_h = nn.Conv1d(ch, ch, 9, padding=4, groups=ch, bias=False)
        self.dw_w = nn.Conv1d(ch, ch, 9, padding=4, groups=ch, bias=False)
        self.gate = nn.Conv2d(ch, ch, 1, bias=True)
        self.out_proj = nn.Conv2d(ch, ch, 1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        z = self.in_proj(x)
        zh = z.permute(0, 3, 1, 2).reshape(b * w, c, h)
        zw = z.permute(0, 2, 1, 3).reshape(b * h, c, w)
        zh = self.dw_h(zh).reshape(b, w, c, h).permute(0, 2, 3, 1)
        zw = self.dw_w(zw).reshape(b, h, c, w).permute(0, 2, 1, 3)
        g = torch.sigmoid(self.gate(x))
        return x + self.out_proj((zh + zw) * g)


class _Block(nn.Module):
    def __init__(self, in_ch, out_ch, use_ssm=False):
        super().__init__()
        self.conv1 = _ReflectConv(in_ch, out_ch)
        self.conv2 = _ReflectConv(out_ch, out_ch)
        self.energy = _TextureEnergyGate(out_ch)
        self.ssm = _SSM2D(out_ch) if use_ssm else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.energy(x)
        x = self.ssm(x)
        return x + r


class texture_energy_modulated_ssm_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.depth = depth
        chs = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev = in_channels
        for i, ch in enumerate(chs):
            self.encoder.append(_Block(prev, ch, use_ssm=(i >= max(0, depth - 3))))
            prev = ch

        self.bottleneck = _Block(chs[-1], chs[-1], use_ssm=True)

        self.decoder = nn.ModuleList()
        self.up_proj = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_proj.append(nn.Conv2d(chs[i + 1], chs[i], 1, bias=False))
            self.decoder.append(_Block(chs[i] * 2, chs[i], use_ssm=(i >= max(0, depth - 3))))

        self.head = nn.Sequential(
            _ReflectConv(chs[0], chs[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(chs[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked
        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i != self.depth - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for proj, block, skip in zip(self.up_proj, self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = proj(y)
            y = block(torch.cat([y, skip], dim=1))

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(y)

        return torch.where(valid, y, torch.full_like(y, float("nan")))


