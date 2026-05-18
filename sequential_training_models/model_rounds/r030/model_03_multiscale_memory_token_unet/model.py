import torch
import torch.nn as nn
import torch.nn.functional as F

def _num_groups(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1

class _ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))

class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv2d(in_channels, out_channels, 3, bias=False)
        self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = _ReflectConv2d(out_channels, out_channels, 3, bias=False)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)

class _MultiScaleMemoryTokenBlock(nn.Module):
    def __init__(self, channels, memory_tokens=16, pool_sizes=(1, 2, 4)):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.memory = nn.Parameter(torch.zeros(1, memory_tokens, channels))
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.pool_sizes = pool_sizes
        self.scale = channels ** -0.5

        nn.init.normal_(self.memory, mean=0.0, std=0.02)

    def forward(self, x):
        residual = x
        z = self.norm(x)
        b, c, h, w = z.shape

        spatial = z.flatten(2).transpose(1, 2)
        pooled_tokens = []
        for size in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(z, (size, size))
            pooled_tokens.append(pooled.flatten(2).transpose(1, 2))

        tokens = torch.cat([self.memory.expand(b, -1, -1)] + pooled_tokens, dim=1)

        q = self.to_q(spatial)
        k = self.to_k(tokens)
        v = self.to_v(tokens)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        context = torch.matmul(attn, v)
        context = self.proj(context).transpose(1, 2).reshape(b, c, h, w)

        return residual + self.gamma * context

class multiscale_memory_token_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = _ConvBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(_ConvBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            _ConvBlock(channels[-1], channels[-1]),
            _MultiScaleMemoryTokenBlock(channels[-1]),
            _ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoder.append(_ConvBlock(channels[i + 1] + channels[i], channels[i]))

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(channels[0], out_channels, 3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x).any(dim=1, keepdim=True)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for block in self.encoder:
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        output = self.out_conv(self.out_pad(y))

        if valid.shape[1] != output.shape[1]:
            valid = valid.expand(-1, output.shape[1], -1, -1)

        output = output.clone()
        output[~valid] = torch.nan
        return output


