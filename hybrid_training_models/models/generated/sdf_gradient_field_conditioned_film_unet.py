import torch
import torch.nn as nn
import torch.nn.functional as F


class sdf_gradient_field_conditioned_film_unet(nn.Module):
    @staticmethod
    def _group_count(channels):
        for groups in range(min(8, channels), 0, -1):
            if channels % groups == 0:
                return groups
        return 1

    class ReflectionConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = sdf_gradient_field_conditioned_film_unet.ReflectionConv2d(in_channels, out_channels, 3)
            self.norm1 = nn.GroupNorm(sdf_gradient_field_conditioned_film_unet._group_count(out_channels), out_channels)
            self.conv2 = sdf_gradient_field_conditioned_film_unet.ReflectionConv2d(out_channels, out_channels, 3)
            self.norm2 = nn.GroupNorm(sdf_gradient_field_conditioned_film_unet._group_count(out_channels), out_channels)
            self.skip = None
            if in_channels != out_channels:
                self.skip = nn.Conv2d(in_channels, out_channels, 1, padding=0)

        def forward(self, x, gamma=None, beta=None):
            residual = x if self.skip is None else self.skip(x)
            x = F.silu(self.norm1(self.conv1(x)))
            x = self.norm2(self.conv2(x))
            if gamma is not None and beta is not None:
                x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            return F.silu(x + residual)

    class FiLM(nn.Module):
        def __init__(self, cond_channels, channels):
            super().__init__()
            hidden = max(cond_channels, channels)
            self.net = nn.Sequential(
                nn.Linear(cond_channels, hidden),
                nn.SiLU(),
                nn.Linear(hidden, channels * 2),
            )

        def forward(self, cond):
            gamma, beta = self.net(cond).chunk(2, dim=1)
            return gamma, beta

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.input_proj = self.ReflectionConv2d(in_channels + 3, channels[0], 3)

        self.encoder = nn.ModuleList()
        for i in range(depth):
            self.encoder.append(self.ConvBlock(channels[i], channels[i]))

        self.downsample = nn.ModuleList()
        for i in range(depth - 1):
            self.downsample.append(self.ReflectionConv2d(channels[i], channels[i + 1], 3, stride=2))

        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.upsample_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.film = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.upsample_proj.append(nn.Conv2d(channels[i + 1], channels[i], 1, padding=0))
            self.decoder.append(self.ConvBlock(channels[i] * 2, channels[i]))
            self.film.append(self.FiLM(4, channels[i]))

        self.out_head = nn.Sequential(
            self.ReflectionConv2d(channels[0], channels[0], 3),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def _gradient_features(self, x):
        left = F.pad(x, (1, 0, 0, 0), mode="reflect")[:, :, :, :-1]
        right = F.pad(x, (0, 1, 0, 0), mode="reflect")[:, :, :, 1:]
        up = F.pad(x, (0, 0, 1, 0), mode="reflect")[:, :, :-1, :]
        down = F.pad(x, (0, 0, 0, 1), mode="reflect")[:, :, 1:, :]

        gx = 0.5 * (right - left)
        gy = 0.5 * (down - up)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)

        if x.shape[1] != 1:
            gx = gx.mean(dim=1, keepdim=True)
            gy = gy.mean(dim=1, keepdim=True)
            grad_mag = grad_mag.mean(dim=1, keepdim=True)

        return torch.cat([gx, gy, grad_mag], dim=1)

    def _conditioning(self, x, grad):
        valid_count = torch.ones_like(x[:, :1]).sum(dim=(2, 3)).clamp_min(1.0)
        mean_h = x[:, :1].sum(dim=(2, 3)) / valid_count
        mean_abs_h = x[:, :1].abs().sum(dim=(2, 3)) / valid_count
        mean_g = grad[:, 2:3].sum(dim=(2, 3)) / valid_count
        max_g = grad[:, 2:3].amax(dim=(2, 3))
        return torch.cat([mean_h, mean_abs_h, mean_g, max_g], dim=1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        grad = self._gradient_features(x_masked)
        cond = self._conditioning(x_masked, grad)

        x0 = torch.cat([x_masked, grad], dim=1)
        h = self.input_proj(x0)

        skips = []
        for i, enc in enumerate(self.encoder):
            h = enc(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.downsample[i](h)

        h = self.bottleneck(h)

        for i, dec in enumerate(self.decoder):
            skip = skips[-(i + 2)]
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = self.upsample_proj[i](h)
            gamma, beta = self.film[i](cond)
            h = dec(torch.cat([h, skip], dim=1), gamma, beta)

        output = self.out_head(h)
        output = output[..., :x.shape[-2], :x.shape[-1]]

        out_valid = valid[:, :1]
        if output.shape[1] != 1:
            out_valid = out_valid.expand(-1, output.shape[1], -1, -1)
        output = torch.where(out_valid, output, torch.full_like(output, float("nan")))
        return output


