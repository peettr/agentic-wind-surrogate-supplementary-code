import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels):
    num_groups = math.gcd(8, num_channels)
    return nn.GroupNorm(num_groups, num_channels)


class channel_transposed_attention_head_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoder = nn.ModuleList()
        prev_ch = in_channels
        for ch in self.channels:
            self.encoder.append(self._conv_block(prev_ch, ch))
            prev_ch = ch

        bottleneck_ch = self.channels[-1]
        self.bottleneck = self._conv_block(bottleneck_ch, bottleneck_ch)

        self.up_projs = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for ch in reversed(self.channels[:-1]):
            self.up_projs.append(nn.Conv2d(prev_ch, ch, kernel_size=1))
            self.decoder.append(self._conv_block(ch * 2, ch))
            prev_ch = ch

        self.attn = self._channel_attention(prev_ch)
        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(prev_ch, prev_ch, kernel_size=3),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(prev_ch, out_channels, kernel_size=3),
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3),
            _gn(out_ch),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3),
            _gn(out_ch),
            nn.GELU(),
        )

    def _channel_attention(self, ch):
        hidden = max(ch // 4, 1)
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked

        for i, block in enumerate(self.encoder):
            h = block(h)
            skips.append(h)
            if i < self.depth - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        h = self.bottleneck(h)

        for up_proj, dec_block, skip in zip(self.up_projs, self.decoder, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = torch.cat([h, skip], dim=1)
            h = dec_block(h)

        h = h * self.attn(h)
        out = self.head(h)

        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_collapsed = valid.all(dim=1, keepdim=True)
        if valid_collapsed.shape[1] != out.shape[1]:
            valid_collapsed = valid_collapsed.expand(-1, out.shape[1], -1, -1)
        return torch.where(valid_collapsed, out, torch.full_like(out, float("nan")))


