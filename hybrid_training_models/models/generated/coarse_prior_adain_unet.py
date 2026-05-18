import torch
import torch.nn as nn
import torch.nn.functional as F

class coarse_prior_adain_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ResBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = coarse_prior_adain_unet.RefConv(in_ch, out_ch)
            self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.conv2 = coarse_prior_adain_unet.RefConv(out_ch, out_ch)
            self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, padding=0)

        def forward(self, x):
            y = F.silu(self.norm1(self.conv1(x)))
            y = self.norm2(self.conv2(y))
            return F.silu(y + self.skip(x))

    class AdaIN(nn.Module):
        def __init__(self, channels, cond_channels):
            super().__init__()
            hidden = max(cond_channels, channels)
            self.net = nn.Sequential(
                nn.Linear(cond_channels, hidden),
                nn.SiLU(),
                nn.Linear(hidden, channels * 2),
            )

        def forward(self, x, cond):
            gamma, beta = self.net(cond).chunk(2, dim=1)
            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.var(dim=(2, 3), unbiased=False, keepdim=True).add(1e-6).sqrt()
            return (x - mean) / std * (1.0 + gamma) + beta

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_block = self.ResBlock(in_channels, channels[0])

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(depth - 1):
            self.downs.append(nn.AvgPool2d(2))
            self.encoders.append(self.ResBlock(channels[i], channels[i + 1]))

        self.coarse_head = nn.Sequential(
            self.RefConv(channels[-1], channels[-1]),
            nn.SiLU(),
            nn.Conv2d(channels[-1], out_channels, 1, padding=0),
        )

        self.bottleneck = self.ResBlock(channels[-1] + out_channels, channels[-1])
        self.adain = self.AdaIN(channels[-1], channels[-1])

        self.up_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_blocks.append(self.ResBlock(channels[i + 1] + channels[i], channels[i]))

        self.out_block = nn.Sequential(
            self.RefConv(channels[0], channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = self.input_block(x_masked)
        skips.append(h)

        for down, enc in zip(self.downs, self.encoders):
            h = down(h)
            h = enc(h)
            skips.append(h)

        coarse = self.coarse_head(h)
        cond = h.mean(dim=(2, 3))
        h = torch.cat([h, coarse], dim=1)
        h = self.bottleneck(h)
        h = self.adain(h, cond)

        for i, up_block in enumerate(self.up_blocks):
            skip = skips[-(i + 2)]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = up_block(h)

        out = self.out_block(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid = valid.expand_as(out)
        out = torch.where(valid, out, torch.full_like(out, float("nan")))
        return out


