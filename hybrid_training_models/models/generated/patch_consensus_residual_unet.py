import torch
import torch.nn as nn
import torch.nn.functional as F


class patch_consensus_residual_unet(nn.Module):
    @staticmethod
    def _gn(channels):
        for groups in (8, 4, 2, 1):
            if channels % groups == 0:
                return nn.GroupNorm(groups, channels)
        return nn.GroupNorm(1, channels)

    class _ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class _ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = patch_consensus_residual_unet._ReflectConv(in_channels, out_channels, 3, bias=False)
            self.norm1 = patch_consensus_residual_unet._gn(out_channels)
            self.conv2 = patch_consensus_residual_unet._ReflectConv(out_channels, out_channels, 3, bias=False)
            self.norm2 = patch_consensus_residual_unet._gn(out_channels)
            self.skip = nn.Identity()
            if in_channels != out_channels:
                self.skip = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

        def forward(self, x):
            residual = self.skip(x)
            x = F.silu(self.norm1(self.conv1(x)), inplace=False)
            x = self.norm2(self.conv2(x))
            return F.silu(x + residual, inplace=False)

    class _PatchConsensus(nn.Module):
        def __init__(self, channels):
            super().__init__()
            hidden = max(4, channels // 4)
            self.reduce = nn.Conv2d(channels * 2, hidden, 1, padding=0)
            self.expand = nn.Conv2d(hidden, channels, 1, padding=0)

        def forward(self, x):
            local = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2, count_include_pad=False)
            pooled = F.adaptive_avg_pool2d(x, output_size=(8, 8))
            pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
            gate = torch.cat([local, pooled], dim=1)
            gate = F.silu(self.reduce(gate), inplace=False)
            gate = torch.sigmoid(self.expand(gate))
            return x * gate + x

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = self._ReflectConv(in_channels, channels[0], 3, bias=False)

        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        prev = channels[0]
        for ch in channels:
            self.encoder.append(self._ResidualBlock(prev, ch))
            self.down.append(nn.AvgPool2d(kernel_size=2, stride=2))
            prev = ch

        self.bottleneck = nn.Sequential(
            self._ResidualBlock(channels[-1], channels[-1]),
            self._PatchConsensus(channels[-1]),
            self._ResidualBlock(channels[-1], channels[-1]),
        )

        self.up_proj = nn.ModuleList()
        self.decoder = nn.ModuleList()
        cur = channels[-1]
        for ch in reversed(channels):
            self.up_proj.append(nn.Conv2d(cur, ch, 1, padding=0, bias=False))
            self.decoder.append(self._ResidualBlock(ch + ch, ch))
            cur = ch

        self.head = nn.Sequential(
            self._ResidualBlock(channels[0], channels[0]),
            self._ReflectConv(channels[0], out_channels, 3, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        original_size = x_masked.shape[-2:]
        x = self.stem(x_masked)

        skips = []
        for enc, down in zip(self.encoder, self.down):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for proj, dec, skip in zip(self.up_proj, self.decoder, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = proj(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        output = self.head(x)
        if output.shape[-2:] != original_size:
            output = F.interpolate(output, size=original_size, mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(output)

        output = output.clone()
        output[~valid] = float("nan")
        return output


