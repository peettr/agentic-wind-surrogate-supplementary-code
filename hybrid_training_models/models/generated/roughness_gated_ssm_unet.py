import torch
import torch.nn as nn
import torch.nn.functional as F

class roughness_gated_ssm_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, groups=groups, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                roughness_gated_ssm_unet.ReflectConv(in_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                roughness_gated_ssm_unet.ReflectConv(out_ch, out_ch, 3),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    class RoughnessGate(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.gate = nn.Sequential(
                roughness_gated_ssm_unet.ReflectConv(ch + 1, ch, 3),
                nn.GroupNorm(min(8, ch), ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(ch, ch, 1),
                nn.Sigmoid(),
            )

        def forward(self, x, h):
            dx = F.pad(h[:, :, :, 1:] - h[:, :, :, :-1], (1, 0, 0, 0), mode="reflect")
            dy = F.pad(h[:, :, 1:, :] - h[:, :, :-1, :], (0, 0, 1, 0), mode="reflect")
            r = torch.sqrt(dx * dx + dy * dy + 1e-6)
            r = F.interpolate(r, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return x * (1.0 + self.gate(torch.cat([x, r], dim=1)))

    class AxialSSM(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.in_proj = nn.Conv2d(ch, ch * 2, 1)
            self.dw_h = nn.Conv1d(ch, ch, 9, padding=4, groups=ch, bias=False)
            self.dw_w = nn.Conv1d(ch, ch, 9, padding=4, groups=ch, bias=False)
            self.out_proj = nn.Conv2d(ch, ch, 1)
            self.norm = nn.GroupNorm(min(8, ch), ch)

        def forward(self, x):
            u, g = self.in_proj(x).chunk(2, dim=1)
            b, c, h, w = u.shape

            yh = u.permute(0, 3, 1, 2).reshape(b * w, c, h)
            yh = self.dw_h(yh).reshape(b, w, c, h).permute(0, 2, 3, 1)

            yw = u.permute(0, 2, 1, 3).reshape(b * h, c, w)
            yw = self.dw_w(yw).reshape(b, h, c, w).permute(0, 2, 1, 3)

            y = (yh + yw) * torch.sigmoid(g)
            return x + self.out_proj(self.norm(y))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self.ConvBlock(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.gates = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(nn.AvgPool2d(2))
            self.encoders.append(self.ConvBlock(channels[i], channels[i + 1]))
            self.gates.append(self.RoughnessGate(channels[i + 1]))

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.AxialSSM(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], 1, bias=False))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0], 3),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        out_valid = valid.all(dim=1, keepdim=True)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h0 = x_masked[:, :1]
        y = self.stem(x_masked)
        skips.append(y)

        for down, enc, gate in zip(self.downs, self.encoders, self.gates):
            y = down(y)
            y = enc(y)
            y = gate(y, h0)
            skips.append(y)

        y = self.bottleneck(y)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = dec(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels != 1:
            out_valid = out_valid.expand(-1, self.out_channels, -1, -1)

        return y.masked_fill(~out_valid, float("nan"))


