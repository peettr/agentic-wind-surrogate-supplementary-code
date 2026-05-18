import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class RefConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class GatedDWFFN(nn.Module):
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.project_in = RefConv2d(channels, hidden * 2, kernel_size=1)
        self.dwconv = RefConv2d(hidden * 2, hidden * 2, kernel_size=3, groups=hidden * 2)
        self.project_out = RefConv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class MDTA(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = RefConv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = RefConv2d(channels * 3, channels * 3, kernel_size=3, groups=channels * 3, bias=False)
        self.project_out = RefConv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        head_dim = c // self.heads

        q = q.reshape(b, self.heads, head_dim, h * w)
        k = k.reshape(b, self.heads, head_dim, h * w)
        v = v.reshape(b, self.heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        return self.project_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        heads = max(1, min(8, channels // 24))
        while channels % heads != 0:
            heads -= 1
        self.norm1 = LayerNorm2d(channels)
        self.attn = MDTA(channels, heads=heads)
        self.norm2 = LayerNorm2d(channels)
        self.ffn = GatedDWFFN(channels)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm2d(channels),
            RefConv2d(channels, channels, kernel_size=3),
            nn.GELU(),
            RefConv2d(channels, channels, kernel_size=3),
        )

    def forward(self, x):
        return x + self.block(x)


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = RefConv2d(in_ch, out_ch, kernel_size=3, stride=2)

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            RefConv2d(in_ch, out_ch * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class restormer_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=24, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.out_channels = out_channels
        self.embed = RefConv2d(in_channels, channels[0], kernel_size=3)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i, ch in enumerate(channels):
            self.encoders.append(nn.Sequential(TransformerBlock(ch), ResBlock(ch)))
            if i < depth - 1:
                self.downs.append(Downsample(ch, channels[i + 1]))

        self.bottleneck = nn.Sequential(
            TransformerBlock(channels[-1]),
            TransformerBlock(channels[-1]),
            ResBlock(channels[-1]),
        )

        self.ups = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.ups.append(Upsample(channels[i + 1], channels[i]))
            self.fuse.append(RefConv2d(channels[i] * 2, channels[i], kernel_size=1))
            self.decoders.append(nn.Sequential(TransformerBlock(channels[i]), ResBlock(channels[i])))

        self.refine = nn.Sequential(
            TransformerBlock(channels[0]),
            ResBlock(channels[0]),
        )
        self.output = RefConv2d(channels[0], out_channels, kernel_size=3)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        size = x_masked.shape[-2:]
        y = self.embed(x_masked)

        skips = []
        for i, enc in enumerate(self.encoders):
            y = enc(y)
            skips.append(y)
            if i < len(self.downs):
                y = self.downs[i](y)

        y = self.bottleneck(y)

        for up, fuse, dec, skip in zip(self.ups, self.fuse, self.decoders, reversed(skips[:-1])):
            y = up(y)
            if y.shape[-2:] != skip.shape[-2:]:
                y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = fuse(torch.cat([y, skip], dim=1))
            y = dec(y)

        y = self.output(self.refine(y))
        if y.shape[-2:] != size:
            y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)

        if valid.shape[1] != y.shape[1]:
            valid_out = valid[:, :1].expand(-1, y.shape[1], -1, -1)
        else:
            valid_out = valid
        y = y.masked_fill(~valid_out, float("nan"))
        return y