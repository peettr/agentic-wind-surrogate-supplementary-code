import torch
import torch.nn as nn
import torch.nn.functional as F

class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels, 3)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels, 3)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)

    def forward(self, x, gamma=None, beta=None):
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if gamma is not None and beta is not None:
            x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return F.silu(x)


class global_descriptor_film_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = _ConvBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(_ConvBlock(channels[i - 1], channels[i]))

        self.bottleneck = _ConvBlock(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(_ConvBlock(channels[i + 1] + channels[i], channels[i]))

        film_channels = channels + [channels[-1]] + list(reversed(channels[:-1]))
        self.film_channels = film_channels
        film_total = sum(film_channels)

        gd_hidden = max(n_c * 4, channels[-1])
        self.global_descriptor = nn.Sequential(
            nn.Linear(channels[-1], gd_hidden),
            nn.SiLU(),
            nn.Linear(gd_hidden, film_total * 2)
        )

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3),
            nn.SiLU(),
            _ReflectConv(channels[0], out_channels, 3)
        )

    def _film_params(self, descriptor):
        film = self.global_descriptor(descriptor)
        gammas, betas = torch.chunk(film, 2, dim=1)

        gamma_parts = []
        beta_parts = []
        start = 0
        for c in self.film_channels:
            end = start + c
            gamma_parts.append(gammas[:, start:end])
            beta_parts.append(betas[:, start:end])
            start = end

        return gamma_parts, beta_parts

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for block in self.encoder:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            h = block(h)
            skips.append(h)

        descriptor = F.adaptive_avg_pool2d(h, 1).flatten(1)
        gamma, beta = self._film_params(descriptor)

        film_idx = 0
        h = self.stem(x_masked, gamma[film_idx], beta[film_idx])
        skips = [h]
        film_idx += 1

        for block in self.encoder:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            h = block(h, gamma[film_idx], beta[film_idx])
            skips.append(h)
            film_idx += 1

        h = self.bottleneck(h, gamma[film_idx], beta[film_idx])
        film_idx += 1

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h, gamma[film_idx], beta[film_idx])
            film_idx += 1

        out = self.head(h)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid_out = valid.any(dim=1, keepdim=True).expand_as(out)
        else:
            valid_out = valid.expand_as(out)

        out = torch.where(valid_out, out, torch.full_like(out, float("nan")))
        return out


