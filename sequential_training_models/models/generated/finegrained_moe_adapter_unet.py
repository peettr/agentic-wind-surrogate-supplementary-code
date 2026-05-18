import torch
import torch.nn as nn
import torch.nn.functional as F

class finegrained_moe_adapter_unet(nn.Module):
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
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class _MoEAdapter(nn.Module):
        def __init__(self, channels, hidden_ratio=4, num_experts=4):
            super().__init__()
            hidden = max(channels // hidden_ratio, 8)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, num_experts, 1),
                nn.Softmax(dim=1),
            )
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(channels, hidden, 1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(hidden, channels, 1),
                )
                for _ in range(num_experts)
            ])
            self.scale = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            weights = self.gate(x)
            y = 0
            for i, expert in enumerate(self.experts):
                y = y + expert(x) * weights[:, i:i + 1]
            return x + self.scale * y

    class _Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.pool = nn.AvgPool2d(2)
            self.block = finegrained_moe_adapter_unet._ConvBlock(in_ch, out_ch)
            self.adapter = finegrained_moe_adapter_unet._MoEAdapter(out_ch)

        def forward(self, x):
            x = self.pool(x)
            x = self.block(x)
            return self.adapter(x)

    class _Up(nn.Module):
        def __init__(self, in_ch, skip_ch, out_ch):
            super().__init__()
            self.reduce = nn.Conv2d(in_ch, out_ch, 1)
            self.block = finegrained_moe_adapter_unet._ConvBlock(out_ch + skip_ch, out_ch)
            self.adapter = finegrained_moe_adapter_unet._MoEAdapter(out_ch)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.reduce(x)
            x = torch.cat([x, skip], dim=1)
            x = self.block(x)
            return self.adapter(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        max_ch = n_c * 8
        channels = [min(n_c * (2 ** i), max_ch) for i in range(depth)]

        self.stem = self._ConvBlock(in_channels, channels[0])
        self.stem_adapter = self._MoEAdapter(channels[0])

        self.encoder = nn.ModuleList([
            self._Down(channels[i - 1], channels[i])
            for i in range(1, depth)
        ])

        self.bottleneck = nn.Sequential(
            self._ConvBlock(channels[-1], channels[-1]),
            self._MoEAdapter(channels[-1]),
        )

        self.decoder = nn.ModuleList([
            self._Up(channels[i], channels[i - 1], channels[i - 1])
            for i in range(depth - 1, 0, -1)
        ])

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], 3, bias=False),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        x = self.stem_adapter(self.stem(x_masked))
        skips.append(x)

        for down in self.encoder:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            x = up(x, skip)

        output = self.head(x)

        if output.shape[-2:] != x_masked.shape[-2:]:
            output = F.interpolate(output, size=x_masked.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid.all(dim=1, keepdim=True).expand_as(output)

        output = output.clone()
        output[~valid] = float("nan")
        return output