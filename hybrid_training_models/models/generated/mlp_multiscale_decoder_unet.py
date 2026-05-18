import torch
import torch.nn as nn
import torch.nn.functional as F

class mlp_multiscale_decoder_unet(nn.Module):
    class _ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class _Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            groups = 1
            for g in (8, 6, 4, 3, 2, 1):
                if out_ch % g == 0:
                    groups = g
                    break

            self.conv1 = mlp_multiscale_decoder_unet._ReflectConv(in_ch, out_ch, 3)
            self.norm1 = nn.GroupNorm(groups, out_ch)
            self.conv2 = mlp_multiscale_decoder_unet._ReflectConv(out_ch, out_ch, 3)
            self.norm2 = nn.GroupNorm(groups, out_ch)
            self.mix = nn.Sequential(
                nn.Conv2d(out_ch, out_ch * 2, 1),
                nn.GELU(),
                nn.Conv2d(out_ch * 2, out_ch, 1),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

        def forward(self, x):
            residual = self.skip(x)
            x = F.gelu(self.norm1(self.conv1(x)))
            x = F.gelu(self.norm2(self.conv2(x)))
            x = self.mix(x) + x
            return F.gelu(x + residual)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(self._Block(prev, ch))
            prev = ch

        self.bottleneck = self._Block(channels[-1], channels[-1])

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_convs.append(nn.Conv2d(channels[i + 1], channels[i], 1))
            self.decoders.append(self._Block(channels[i] * 2, channels[i]))

        self.final = nn.Sequential(
            self._ReflectConv(channels[0], channels[0], 3),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up, decoder, skip in zip(self.up_convs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up(h)
            h = decoder(torch.cat([h, skip], dim=1))

        out = self.final(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid
        if valid_out.shape[1] != self.out_channels:
            valid_out = valid_out[:, :1].expand(-1, self.out_channels, -1, -1)

        return torch.where(valid_out, out, torch.full_like(out, float("nan")))


