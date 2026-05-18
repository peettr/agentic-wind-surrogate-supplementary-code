import torch
import torch.nn as nn
import torch.nn.functional as F

class derived_feature_aggregate_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                derived_feature_aggregate_unet.RefConv(in_ch, out_ch),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                derived_feature_aggregate_unet.RefConv(out_ch, out_ch),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.block = derived_feature_aggregate_unet.ConvBlock(in_ch + skip_ch, out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.block(torch.cat([x, skip], dim=1))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        feature_channels = in_channels * 5

        self.register_buffer("sobel_x", torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
        ).unsqueeze(0), persistent=False)
        self.register_buffer("sobel_y", torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
        ).unsqueeze(0), persistent=False)
        self.register_buffer("laplace", torch.tensor(
            [[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]
        ).unsqueeze(0), persistent=False)

        self.feature_pad = nn.ReflectionPad2d(1)

        self.encoders = nn.ModuleList()
        prev_ch = feature_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self.UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            self.RefConv(channels[0], channels[0]),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    def _derived_features(self, x):
        c = x.shape[1]
        x_pad = self.feature_pad(x)

        sobel_x = self.sobel_x.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        laplace = self.laplace.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)

        gx = F.conv2d(x_pad, sobel_x, groups=c) / 8.0
        gy = F.conv2d(x_pad, sobel_y, groups=c) / 8.0
        lap = F.conv2d(x_pad, laplace, groups=c)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)

        return torch.cat([x, gx, gy, lap, grad_mag], dim=1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        y = self._derived_features(x_masked)

        skips = []
        for i, encoder in enumerate(self.encoders):
            y = encoder(y)
            skips.append(y)
            if i < self.depth - 1:
                y = self.pool(y)

        y = self.bottleneck(y)

        skips = skips[:-1][::-1]
        for decoder, skip in zip(self.decoders, skips):
            y = decoder(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != y.shape[1]:
            valid_out = valid_out[:, :1].expand(-1, y.shape[1], -1, -1)

        y = torch.where(valid_out, y, torch.full_like(y, float("nan")))
        return y