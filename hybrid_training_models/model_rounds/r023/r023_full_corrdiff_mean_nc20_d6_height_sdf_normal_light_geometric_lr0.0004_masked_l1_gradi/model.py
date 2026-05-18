import torch
import torch.nn as nn
import torch.nn.functional as F

class corrdiff_mean(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()

        def channels_at(i):
            return min(n_c * (2 ** i), n_c * 8)

        class ConvBlock(nn.Module):
            def __init__(self, c_in, c_out):
                super().__init__()
                groups = min(8, c_out)
                while c_out % groups != 0:
                    groups -= 1
                self.net = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(c_in, c_out, kernel_size=3),
                    nn.GroupNorm(groups, c_out),
                    nn.SiLU(inplace=True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(c_out, c_out, kernel_size=3),
                    nn.GroupNorm(groups, c_out),
                    nn.SiLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)

        depth = max(1, int(depth))
        channels = [channels_at(i) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        c_in = in_channels
        for i, c_out in enumerate(channels):
            self.encoders.append(ConvBlock(c_in, c_out))
            if i < depth - 1:
                self.downs.append(
                    nn.Sequential(
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(c_out, channels[i + 1], kernel_size=3, stride=2),
                        nn.SiLU(inplace=True),
                    )
                )
            c_in = channels[i + 1] if i < depth - 1 else c_out

        self.bottleneck = ConvBlock(channels[-1], channels[-1])

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1))
            self.decoders.append(ConvBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x[:, :1])
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        h = x_masked

        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        h = self.bottleneck(h)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        output = self.head(h)
        output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)
        output = torch.where(valid.expand_as(output), output, torch.full_like(output, float("nan")))
        return output