import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class FiLM(nn.Module):
    def __init__(self, token_dim, channels):
        super().__init__()
        hidden = max(token_dim, channels)
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )

    def forward(self, x, token):
        gamma, beta = self.net(token).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + gamma) + beta


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, token_dim):
        super().__init__()
        groups1 = min(8, out_channels)
        groups2 = min(8, out_channels)

        while out_channels % groups1 != 0:
            groups1 -= 1
        while out_channels % groups2 != 0:
            groups2 -= 1

        self.conv1 = ReflectConv(in_channels, out_channels)
        self.norm1 = nn.GroupNorm(groups1, out_channels)
        self.film1 = FiLM(token_dim, out_channels)

        self.conv2 = ReflectConv(out_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups2, out_channels)
        self.film2 = FiLM(token_dim, out_channels)

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, token):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film1(h, token)
        h = F.silu(h, inplace=True)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.film2(h, token)

        return F.silu(h + self.skip(x), inplace=True)


class boundary_token_film_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        max_channels = n_c * 8
        channels = [min(n_c * (2 ** i), max_channels) for i in range(depth)]
        self.channels = channels

        token_dim = max(32, n_c * 4)
        self.boundary_token = nn.Sequential(
            nn.Linear(in_channels * 4, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(inplace=True),
        )

        self.stem = ResBlock(in_channels, channels[0], token_dim)

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(ResBlock(channels[i - 1], channels[i], token_dim))

        self.bottleneck = nn.Sequential(
            ResBlock(channels[-1], channels[-1], token_dim),
            ResBlock(channels[-1], channels[-1], token_dim),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(ResBlock(channels[i + 1] + channels[i], channels[i], token_dim))

        self.head = nn.Sequential(
            ReflectConv(channels[0], channels[0]),
            nn.SiLU(inplace=True),
            ReflectConv(channels[0], out_channels, kernel_size=3, bias=True),
        )

    def _boundary_token(self, x, valid):
        count_top = valid[:, :, 0, :].sum(dim=2).clamp_min(1.0)
        count_bottom = valid[:, :, -1, :].sum(dim=2).clamp_min(1.0)
        count_left = valid[:, :, :, 0].sum(dim=2).clamp_min(1.0)
        count_right = valid[:, :, :, -1].sum(dim=2).clamp_min(1.0)

        top = x[:, :, 0, :].sum(dim=2) / count_top
        bottom = x[:, :, -1, :].sum(dim=2) / count_bottom
        left = x[:, :, :, 0].sum(dim=2) / count_left
        right = x[:, :, :, -1].sum(dim=2) / count_right

        token = torch.cat([top, bottom, left, right], dim=1)
        return self.boundary_token(token)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))
        token = self._boundary_token(x_masked, valid.to(x_masked.dtype))

        skips = []
        h = self.stem(x_masked, token)
        skips.append(h)

        for block in self.encoder:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            h = block(h, token)
            skips.append(h)

        for block in self.bottleneck:
            h = block(h, token)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h, token)

        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid_out = valid[:, :1].expand(-1, out.shape[1], -1, -1)
        else:
            valid_out = valid

        return torch.where(valid_out, out, torch.full_like(out, float("nan")))