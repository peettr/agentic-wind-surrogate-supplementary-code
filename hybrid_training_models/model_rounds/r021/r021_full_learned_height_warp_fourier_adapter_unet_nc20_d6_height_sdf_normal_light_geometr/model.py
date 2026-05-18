import torch
import torch.nn as nn
import torch.nn.functional as F

class learned_height_warp_fourier_adapter_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, groups=8):
            super().__init__()
            pad = kernel_size // 2
            g = min(groups, out_ch)
            while out_ch % g != 0:
                g -= 1
            self.net = nn.Sequential(
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=False),
                nn.GroupNorm(g, out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                learned_height_warp_fourier_adapter_unet.ReflectConv(in_ch, out_ch),
                learned_height_warp_fourier_adapter_unet.ReflectConv(out_ch, out_ch),
            )
            self.proj = None
            if in_ch != out_ch:
                self.proj = nn.Conv2d(in_ch, out_ch, 1, padding=0, bias=False)

        def forward(self, x):
            y = self.net(x)
            if self.proj is not None:
                x = self.proj(x)
            return y + x

    class FourierAdapter(nn.Module):
        def __init__(self, channels, modes=24):
            super().__init__()
            self.modes = modes
            self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.mix = nn.Conv2d(channels, channels, 1, padding=0, bias=True)

        def forward(self, x):
            b, c, h, w = x.shape
            mh = min(self.modes, h)
            mw = min(self.modes, w // 2 + 1)

            xf = torch.fft.rfft2(x.float(), norm="ortho")
            low = torch.zeros_like(xf)
            low[:, :, :mh, :mw] = xf[:, :, :mh, :mw]
            if mh > 1:
                low[:, :, -mh + 1:, :mw] = xf[:, :, -mh + 1:, :mw]

            y = torch.fft.irfft2(low, s=(h, w), norm="ortho").to(dtype=x.dtype)
            return x + self.mix(y) * self.scale

    class HeightWarp(nn.Module):
        def __init__(self, channels):
            super().__init__()
            hidden = max(8, channels // 2)
            self.net = nn.Sequential(
                learned_height_warp_fourier_adapter_unet.ReflectConv(1, hidden),
                nn.ReflectionPad2d(1),
                nn.Conv2d(hidden, 2, 3, padding=0),
                nn.Tanh(),
            )
            self.strength = nn.Parameter(torch.tensor(0.02))

        def forward(self, feat, height):
            h, w = feat.shape[-2:]
            height = F.interpolate(height, size=(h, w), mode="bilinear", align_corners=False)
            flow = self.net(height).permute(0, 2, 3, 1) * self.strength

            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype),
                torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype),
                indexing="ij",
            )
            grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(feat.shape[0], -1, -1, -1)
            return F.grid_sample(
                feat,
                grid + flow,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True,
            )

    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = self.ConvBlock(in_channels, channels[0])
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(nn.AvgPool2d(2))
            self.encoders.append(self.ConvBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.FourierAdapter(channels[-1]),
            self.ConvBlock(channels[-1], channels[-1]),
        )

        self.warps = nn.ModuleList([self.HeightWarp(ch) for ch in channels[:-1]])

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(self.ReflectConv(channels[i + 1], channels[i], kernel_size=3))
            self.decoders.append(self.ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            self.ReflectConv(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))
        height = x_masked[:, :1]

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for down, enc in zip(self.downs, self.encoders):
            y = down(y)
            y = enc(y)
            skips.append(y)

        y = self.bottleneck(y)

        for i, (up, dec) in enumerate(zip(self.up_convs, self.decoders)):
            skip = skips[-(i + 2)]
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(y)
            skip = self.warps[-(i + 1)](skip, height)
            y = dec(torch.cat([y, skip], dim=1))

        out = self.head(y)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels == self.in_channels:
            out_valid = valid
        else:
            out_valid = valid[:, :1].expand(-1, self.out_channels, -1, -1)

        return torch.where(out_valid, out, torch.full_like(out, float("nan")))


