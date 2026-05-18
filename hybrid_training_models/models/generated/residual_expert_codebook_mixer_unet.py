import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv1 = _ReflectConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.silu(y + self.skip(x))


class _ExpertCodebookMixer(nn.Module):
    def __init__(self, channels, experts=4, codewords=16):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.codebook = nn.Parameter(torch.randn(codewords, channels) * 0.02)
        self.query = nn.Linear(channels, codewords, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, experts)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.GroupNorm(min(8, channels), channels),
                nn.SiLU()
            )
            for _ in range(experts)
        ])
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        code_weights = F.softmax(self.query(pooled), dim=1)
        code_context = code_weights @ self.codebook
        gates = F.softmax(self.gate(torch.cat([pooled, code_context], dim=1)), dim=1)

        y = 0
        for i, expert in enumerate(self.experts):
            y = y + expert(x) * gates[:, i].view(-1, 1, 1, 1)

        return x + self.proj(y)


class residual_expert_codebook_mixer_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = _ResidualBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(depth):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            self.encoder.append(_ResidualBlock(in_ch, out_ch))

        self.pool = nn.AvgPool2d(2)

        self.bottleneck = nn.Sequential(
            _ResidualBlock(channels[-1], channels[-1]),
            _ExpertCodebookMixer(channels[-1]),
            _ResidualBlock(channels[-1], channels[-1])
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(nn.Sequential(
                _ResidualBlock(channels[i + 1] + channels[i], channels[i]),
                _ExpertCodebookMixer(channels[i]),
                _ResidualBlock(channels[i], channels[i])
            ))

        self.head = nn.Sequential(
            _ResidualBlock(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, 3, padding=0)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        y = self.stem(x_masked)
        skips = []

        for i, block in enumerate(self.encoder):
            y = block(y)
            skips.append(y)
            if i != self.depth - 1:
                y = self.pool(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))

        y = self.head(y)

        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid.all(dim=1, keepdim=True).expand_as(y)
        return torch.where(out_valid, y, torch.full_like(y, float("nan")))


