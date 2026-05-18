import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class _ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = _ReflectConv(in_channels, out_channels, 3, bias=False)
        self.norm1 = _gn(out_channels)
        self.conv2 = _ReflectConv(out_channels, out_channels, 3, bias=False)
        self.norm2 = _gn(out_channels)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(y + self.skip(x))


class _LatentCrossAttention(nn.Module):
    def __init__(self, channels, latent_tokens=64, heads=4):
        super().__init__()
        heads = max(1, min(heads, channels // 16 if channels >= 16 else 1))
        while channels % heads != 0:
            heads -= 1
        self.latents = nn.Parameter(torch.randn(1, latent_tokens, channels) * 0.02)
        self.to_q_latent = nn.Linear(channels, channels, bias=False)
        self.to_k_field = nn.Linear(channels, channels, bias=False)
        self.to_v_field = nn.Linear(channels, channels, bias=False)
        self.to_q_field = nn.Linear(channels, channels, bias=False)
        self.to_k_latent = nn.Linear(channels, channels, bias=False)
        self.to_v_latent = nn.Linear(channels, channels, bias=False)
        self.proj_latent = nn.Linear(channels, channels)
        self.proj_field = nn.Linear(channels, channels)
        self.norm_field = nn.LayerNorm(channels)
        self.norm_latent = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.heads = heads
        self.scale = (channels // heads) ** -0.5

    def _attention(self, q, k, v):
        b, nq, c = q.shape
        nk = k.shape[1]
        h = self.heads
        d = c // h
        q = q.view(b, nq, h, d).transpose(1, 2)
        k = k.view(b, nk, h, d).transpose(1, 2)
        v = v.view(b, nk, h, d).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)
        return y.transpose(1, 2).contiguous().view(b, nq, c)

    def forward(self, x):
        b, c, h, w = x.shape
        field = x.flatten(2).transpose(1, 2)
        field_n = self.norm_field(field)
        latents = self.latents.expand(b, -1, -1)
        latents_n = self.norm_latent(latents)

        latents = latents + self.proj_latent(self._attention(
            self.to_q_latent(latents_n),
            self.to_k_field(field_n),
            self.to_v_field(field_n),
        ))
        latents_n = self.norm_latent(latents)

        field = field + self.proj_field(self._attention(
            self.to_q_field(field_n),
            self.to_k_latent(latents_n),
            self.to_v_latent(latents_n),
        ))
        field = field + self.ffn(field)
        return field.transpose(1, 2).view(b, c, h, w)


class latent_cross_attention_operator_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = max(1, int(depth))

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(self.depth)]
        self.channels = channels

        self.stem = _ResBlock(in_channels, channels[0])
        self.encoder = nn.ModuleList()
        for i in range(1, self.depth):
            self.encoder.append(_ResBlock(channels[i - 1], channels[i]))

        self.pool = nn.AvgPool2d(2)
        self.bottleneck = nn.Sequential(
            _ResBlock(channels[-1], channels[-1]),
            _LatentCrossAttention(channels[-1]),
            _ResBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(self.depth - 2, -1, -1):
            self.decoder.append(_ResBlock(channels[i + 1] + channels[i], channels[i]))

        self.head = nn.Sequential(
            _ReflectConv(channels[0], channels[0], 3, bias=False),
            _gn(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 1, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = self.stem(x_masked)
        skips.append(y)

        for block in self.encoder:
            y = self.pool(y)
            y = block(y)
            skips.append(y)

        y = self.bottleneck(y)

        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = torch.cat([y, skip], dim=1)
            y = block(y)

        y = self.head(y)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid.all(dim=1, keepdim=True).expand_as(y)
        else:
            valid_out = valid

        return torch.where(valid_out, y, torch.full_like(y, float("nan")))