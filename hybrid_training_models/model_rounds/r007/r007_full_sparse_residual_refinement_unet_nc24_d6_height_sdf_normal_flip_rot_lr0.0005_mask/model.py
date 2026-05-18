import torch
import torch.nn as nn
import torch.nn.functional as F

class sparse_residual_refinement_unet(nn.Module):
    class _ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(out_ch, out_ch, 3, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))

    class _Down(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(ch, ch, 3, stride=2, bias=False),
                nn.GroupNorm(min(8, ch), ch),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class _Up(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.block = sparse_residual_refinement_unet._ConvBlock(out_ch + skip_ch, out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            return self.block(torch.cat([x, skip], dim=1))

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.input_proj = self._ConvBlock(in_channels, channels[0])

        self.encoder_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for i in range(depth):
            self.encoder_blocks.append(self._ConvBlock(channels[i], channels[i]))
            if i < depth - 1:
                self.down_blocks.append(self._Down(channels[i]))
                if channels[i] != channels[i + 1]:
                    self.down_blocks.append(nn.Conv2d(channels[i], channels[i + 1], 1, bias=False))

        self.bottleneck = nn.Sequential(
            self._ConvBlock(channels[-1], channels[-1]),
            self._ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder_blocks.append(self._Up(channels[i + 1], channels[i], channels[i]))

        self.refine = nn.Sequential(
            self._ConvBlock(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

        self.residual_head = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        h = self.input_proj(x_masked)
        skips = []
        down_idx = 0

        for i, enc in enumerate(self.encoder_blocks):
            h = enc(h)
            skips.append(h)
            if i < self.depth - 1:
                h = self.down_blocks[down_idx](h)
                down_idx += 1
                if down_idx < len(self.down_blocks) and isinstance(self.down_blocks[down_idx], nn.Conv2d):
                    h = self.down_blocks[down_idx](h)
                    down_idx += 1

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoder_blocks, reversed(skips[:-1])):
            h = dec(h, skip)

        out = self.refine(h) + self.residual_head(x_masked)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != out.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(out)

        return torch.where(valid, out, torch.full_like(out, float("nan")))


