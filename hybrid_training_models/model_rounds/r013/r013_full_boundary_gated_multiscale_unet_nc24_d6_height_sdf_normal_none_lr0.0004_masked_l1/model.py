import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    if num_channels <= 1:
        return nn.Identity()
    for num_groups in range(min(8, num_channels), 0, -1):
        if num_channels % num_groups == 0 and num_channels // num_groups >= 2:
            return nn.GroupNorm(num_groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=False)
        self.norm = _gn(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        if self.pad > 0:
            h, w = x.shape[-2:]
            pad_h = min(self.pad, max(h - 1, 0))
            pad_w = min(self.pad, max(w - 1, 0))

            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")

            eff_h = 2 * pad_h + 1
            eff_w = 2 * pad_w + 1
            if eff_h != self.kernel_size or eff_w != self.kernel_size:
                start_h = (self.kernel_size - eff_h) // 2
                start_w = (self.kernel_size - eff_w) // 2
                weight = self.conv.weight[
                    :, :, start_h:start_h + eff_h, start_w:start_w + eff_w
                ]
                x = F.conv2d(x, weight, self.conv.bias, padding=0)
            else:
                x = self.conv(x)
        else:
            x = self.conv(x)

        return self.act(self.norm(x))


class _Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels)
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.skip(x)


class _BoundaryGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 8)
        self.gate = nn.Sequential(
            _ReflectConv(1, hidden),
            nn.Conv2d(hidden, channels, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, feat, mask):
        m = F.interpolate(mask.float(), size=feat.shape[-2:], mode="nearest")
        h, w = m.shape[-2:]
        pad_h = 1 if h > 1 else 0
        pad_w = 1 if w > 1 else 0

        if pad_h > 0 or pad_w > 0:
            inv = F.pad(1.0 - m, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
            neigh = F.max_pool2d(inv, kernel_size=(2 * pad_h + 1, 2 * pad_w + 1), stride=1)
        else:
            neigh = torch.zeros_like(m)

        boundary = torch.clamp(neigh * m, 0.0, 1.0)
        return feat * (1.0 + self.gate(boundary))


class boundary_gated_multiscale_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        self.gates = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(_Block(prev, ch))
            self.gates.append(_BoundaryGate(ch))
            prev = ch

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = _Block(channels[-1], channels[-1])

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0, bias=False))
            self.decoders.append(_Block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            h = self.gates[i](h, valid)
            skips.append(h)
            if i < self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 2)]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = self.up_projs[i](h)
            h = dec(torch.cat([h, skip], dim=1))

        out = self.head(h)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if out.shape[1] == valid.shape[1]:
            out_valid = valid
        else:
            out_valid = valid.expand(-1, out.shape[1], -1, -1)
        return torch.where(out_valid, out, torch.full_like(out, float("nan")))


