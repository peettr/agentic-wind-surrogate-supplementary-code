import torch
import torch.nn as nn
import torch.nn.functional as F

class coarse_interp_residual_head_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        def conv3x3(cin, cout):
            return nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(cin, cout, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, cout), cout),
                nn.SiLU(inplace=True),
            )

        class ResBlock(nn.Module):
            def __init__(self, cin, cout):
                super().__init__()
                self.conv1 = conv3x3(cin, cout)
                self.conv2 = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(cout, cout, kernel_size=3, padding=0, bias=False),
                    nn.GroupNorm(min(8, cout), cout),
                )
                self.skip = (
                    nn.Conv2d(cin, cout, kernel_size=1, padding=0, bias=False)
                    if cin != cout else nn.Identity()
                )
                self.act = nn.SiLU(inplace=True)

            def forward(self, x):
                return self.act(self.conv2(self.conv1(x)) + self.skip(x))

        self.encoders = nn.ModuleList()
        prev_c = in_channels
        for c in channels:
            self.encoders.append(ResBlock(prev_c, c))
            prev_c = c

        self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            ResBlock(channels[-1], channels[-1]),
            ResBlock(channels[-1], channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(ResBlock(channels[i + 1] + channels[i], channels[i]))

        self.residual_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

        self.coarse_head = nn.Conv2d(channels[-1], out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != self.depth - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        coarse = self.coarse_head(h)
        coarse = F.interpolate(coarse, size=x.shape[-2:], mode="bilinear", align_corners=False)

        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        out = coarse + self.residual_head(h)

        if self.out_channels == self.in_channels:
            out_valid = valid
        else:
            out_valid = valid[:, :1].expand(-1, self.out_channels, -1, -1)

        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out