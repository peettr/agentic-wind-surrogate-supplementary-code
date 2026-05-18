import torch
import torch.nn as nn
import torch.nn.functional as F


class differentiable_stencil_residual_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = []
        for i in range(depth):
            channels.append(min(n_c * (2 ** i), n_c * 8))
        self.channels = channels

        self.stencil_pad = nn.ReflectionPad2d(1)

        stencil = torch.zeros(4, 1, 3, 3)
        stencil[0, 0, 1, 1] = -4.0
        stencil[0, 0, 0, 1] = 1.0
        stencil[0, 0, 2, 1] = 1.0
        stencil[0, 0, 1, 0] = 1.0
        stencil[0, 0, 1, 2] = 1.0
        stencil[1, 0, 1, 0] = -0.5
        stencil[1, 0, 1, 2] = 0.5
        stencil[2, 0, 0, 1] = -0.5
        stencil[2, 0, 2, 1] = 0.5
        stencil[3, 0, 0, 0] = 0.25
        stencil[3, 0, 0, 2] = -0.25
        stencil[3, 0, 2, 0] = -0.25
        stencil[3, 0, 2, 2] = 0.25
        self.register_buffer("stencil_kernel", stencil.repeat(in_channels, 1, 1, 1))

        self.input_proj = self._conv_block(in_channels * 5, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(depth - 1):
            self.encoder.append(self._conv_block(channels[i], channels[i + 1]))

        self.bottleneck = self._conv_block(channels[-1], channels[-1])

        self.decoder = nn.ModuleList()
        self.up_proj = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up_proj.append(self._conv1x1(channels[i + 1], channels[i]))
            self.decoder.append(self._conv_block(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

        self.residual_head = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def _num_groups(self, channels):
        for groups in range(min(8, channels), 0, -1):
            if channels % groups == 0:
                return groups
        return 1

    def _conv1x1(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0),
            nn.GELU(),
        )

    def _conv_block(self, in_ch, out_ch):
        groups = self._num_groups(out_ch)
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def _stencil_features(self, x):
        s = F.conv2d(
            self.stencil_pad(x),
            self.stencil_kernel,
            padding=0,
            groups=self.in_channels,
        )
        return torch.cat([x, s], dim=1)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        feats = self._stencil_features(x_masked)
        y = self.input_proj(feats)

        skips = [y]
        for block in self.encoder:
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for up, block, skip in zip(self.up_proj, self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up(y)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        y = self.head(y) + self.residual_head(x_masked)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if self.out_channels == self.in_channels:
            valid_out = valid
        else:
            valid_out = valid[:, :1].expand(-1, self.out_channels, -1, -1)

        return y.masked_fill(~valid_out, float("nan"))


