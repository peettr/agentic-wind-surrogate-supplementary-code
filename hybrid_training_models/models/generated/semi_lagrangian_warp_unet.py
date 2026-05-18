import torch
import torch.nn as nn
import torch.nn.functional as F


class semi_lagrangian_warp_unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.net(x) + self.skip(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        max_ch = n_c * 8
        channels = [min(n_c * (2 ** i), max_ch) for i in range(depth)]

        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoders.append(self.ConvBlock(prev_ch, ch))
            prev_ch = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.flow_heads = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            up_ch = channels[i + 1]
            skip_ch = channels[i]
            self.flow_heads.append(nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(up_ch, 2, kernel_size=3, padding=0)
            ))
            self.decoders.append(self.ConvBlock(up_ch + skip_ch, skip_ch))

        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=0),
            nn.SiLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0)
        )

    def _warp(self, x, flow):
        b, c, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, h, w, 2)
        scale = torch.tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)], device=x.device, dtype=x.dtype)
        grid = grid + flow.permute(0, 2, 3, 1) * scale
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = x_masked
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.pool(h)

        h = self.bottleneck(h)

        for flow_head, dec, skip in zip(self.flow_heads, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            flow = torch.tanh(flow_head(h)) * 2.0
            skip = self._warp(skip, flow)
            h = dec(torch.cat([h, skip], dim=1))

        output = self.out(h)
        output_valid = valid if valid.shape[1] == output.shape[1] else valid.all(dim=1, keepdim=True)
        output = torch.where(output_valid, output, torch.full_like(output, float("nan")))
        return output


