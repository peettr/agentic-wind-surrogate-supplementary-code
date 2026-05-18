import torch
import torch.nn as nn
import torch.nn.functional as F

class local_enhanced_selective_scan_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            groups = min(8, out_ch)
            while out_ch % groups != 0:
                groups -= 1

            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, bias=False),
                nn.GroupNorm(groups, out_ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class LocalSelectiveScanBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            groups = min(8, ch)
            while ch % groups != 0:
                groups -= 1

            self.local = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch, kernel_size=3, groups=ch, bias=False),
                nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                nn.GroupNorm(groups, ch),
                nn.SiLU(inplace=True),
            )

            self.gate = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch, kernel_size=3, groups=ch, bias=False),
                nn.Conv2d(ch, ch, kernel_size=1),
                nn.Sigmoid(),
            )

            self.mix_h = nn.Conv1d(ch, ch, kernel_size=1, groups=1, bias=False)
            self.mix_w = nn.Conv1d(ch, ch, kernel_size=1, groups=1, bias=False)

            self.proj = nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=1, bias=False),
                nn.GroupNorm(groups, ch),
            )

        def forward(self, x):
            b, c, h, w = x.shape

            local = self.local(x)
            gate = self.gate(x)

            scan_h = local.mean(dim=3)
            scan_w = local.mean(dim=2)

            scan_h = self.mix_h(scan_h).unsqueeze(3).expand(b, c, h, w)
            scan_w = self.mix_w(scan_w).unsqueeze(2).expand(b, c, h, w)

            y = local + gate * (scan_h + scan_w)
            return F.silu(x + self.proj(y), inplace=True)

    class UpBlock(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            self.conv = local_enhanced_selective_scan_unet.ConvBlock(out_ch + skip_ch, out_ch)
            self.scan = local_enhanced_selective_scan_unet.LocalSelectiveScanBlock(out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            x = self.conv(x)
            return self.scan(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=5):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = []
        for i in range(depth):
            channels.append(min(n_c * (2 ** i), n_c * 8))

        self.encoders = nn.ModuleList()
        self.scans = nn.ModuleList()

        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            self.scans.append(self.LocalSelectiveScanBlock(ch))
            prev_ch = ch

        self.bottleneck = nn.Sequential(
            self.ConvBlock(channels[-1], channels[-1]),
            self.LocalSelectiveScanBlock(channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(self.UpBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, kernel_size=1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = x_masked

        for i, (encoder, scan) in enumerate(zip(self.encoders, self.scans)):
            y = scan(encoder(y))
            skips.append(y)
            if i < self.depth - 1:
                y = F.avg_pool2d(y, kernel_size=2, stride=2)

        y = self.bottleneck(y)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            y = decoder(y, skip)

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid.all(dim=1, keepdim=True).expand_as(y)
        else:
            valid_out = valid

        return torch.where(valid_out, y, torch.full_like(y, float("nan")))


