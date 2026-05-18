import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class beno_film(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        class ReflectConv(nn.Module):
            def __init__(self, c_in, c_out, kernel_size=3, stride=1):
                super().__init__()
                pad = kernel_size // 2
                self.net = nn.Sequential(
                    nn.ReflectionPad2d(pad),
                    nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=0, bias=False),
                    _gn(c_out),
                    nn.SiLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)

        class ResBlock(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.conv1 = ReflectConv(c, c)
                self.conv2 = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(c, c, 3, padding=0, bias=False),
                    _gn(c),
                )
                self.act = nn.SiLU(inplace=True)

            def forward(self, x):
                return self.act(x + self.conv2(self.conv1(x)))

        class DownBlock(nn.Module):
            def __init__(self, c_in, c_out):
                super().__init__()
                self.down = ReflectConv(c_in, c_out, stride=2)
                self.res = ResBlock(c_out)

            def forward(self, x):
                return self.res(self.down(x))

        class UpBlock(nn.Module):
            def __init__(self, c_in, skip_c, c_out):
                super().__init__()
                self.proj = ReflectConv(c_in + skip_c, c_out)
                self.res = ResBlock(c_out)

            def forward(self, x, skip):
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
                return self.res(self.proj(x))

        self.stem = nn.Sequential(
            ReflectConv(in_channels, channels[0]),
            ResBlock(channels[0]),
        )

        self.encoder = nn.ModuleList([
            DownBlock(channels[i], channels[i + 1])
            for i in range(depth - 1)
        ])

        self.bottleneck = nn.Sequential(
            ResBlock(channels[-1]),
            ResBlock(channels[-1]),
        )

        self.film_pool = nn.AdaptiveAvgPool2d(1)
        self.film = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] * 2),
            nn.SiLU(inplace=True),
            nn.Linear(channels[-1] * 2, channels[-1] * 2),
        )

        self.decoder = nn.ModuleList([
            UpBlock(channels[i + 1], channels[i], channels[i])
            for i in range(depth - 2, -1, -1)
        ])

        self.head = nn.Sequential(
            ReflectConv(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for down in self.encoder:
            h = down(h)
            skips.append(h)

        h = self.bottleneck(h)

        film = self.film(self.film_pool(h).flatten(1))
        gamma, beta = film.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        h = h * (1.0 + gamma) + beta

        skips = skips[:-1][::-1]
        for up, skip in zip(self.decoder, skips):
            h = up(h, skip)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid_out = valid.all(dim=1, keepdim=True).expand_as(out)
        else:
            valid_out = valid.expand_as(out)

        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


