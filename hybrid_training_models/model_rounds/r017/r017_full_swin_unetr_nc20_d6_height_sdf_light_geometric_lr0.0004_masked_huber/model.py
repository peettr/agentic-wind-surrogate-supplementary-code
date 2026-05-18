import torch
import torch.nn as nn
import torch.nn.functional as F

class ReflectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        self.block = nn.Sequential(
            ReflectConv2d(in_channels, out_channels, 3, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            ReflectConv2d(out_channels, out_channels, 3, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )

        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)

    def forward(self, x):
        return F.gelu(self.block(x) + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = ReflectConv2d(in_channels, out_channels, 3, 2, bias=False)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        return self.conv(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = ReflectConv2d(in_channels, out_channels, 3, 1, bias=False)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class WindowAttentionBlock(nn.Module):
    def __init__(self, channels, window_size=10, heads=4):
        super().__init__()
        self.channels = channels
        self.window_size = window_size

        heads = min(heads, channels)
        while channels % heads != 0:
            heads -= 1
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, padding=0, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, padding=0, bias=False)

        self.norm2 = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            ReflectConv2d(channels, channels * 2, 3, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1, padding=0, bias=False),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        ws = min(self.window_size, h, w)

        y = self.norm1(x)
        qkv = self.qkv(y)

        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            qkv = F.pad(qkv, (0, pad_w, 0, pad_h), mode="reflect")

        hp, wp = qkv.shape[-2:]
        qkv = qkv.view(b, 3, self.heads, self.head_dim, hp // ws, ws, wp // ws, ws)
        qkv = qkv.permute(1, 0, 4, 6, 2, 5, 7, 3).contiguous()
        qkv = qkv.view(3, b * (hp // ws) * (wp // ws), self.heads, ws * ws, self.head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        y = torch.matmul(attn, v)

        y = y.view(b, hp // ws, wp // ws, self.heads, ws, ws, self.head_dim)
        y = y.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
        y = y.view(b, c, hp, wp)

        if pad_h or pad_w:
            y = y[:, :, :h, :w]

        x = x + self.proj(y)
        x = x + self.ffn(self.norm2(x))
        return x


class swin_unetr(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=20, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.stem = ConvBlock(in_channels, channels[0])

        self.encoder = nn.ModuleList()
        for i in range(1, depth):
            self.encoder.append(DownBlock(channels[i - 1], channels[i]))

        self.bottleneck = nn.Sequential(
            ConvBlock(channels[-1], channels[-1]),
            WindowAttentionBlock(channels[-1]),
            ConvBlock(channels[-1], channels[-1]),
        )

        self.decoder = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.decoder.append(UpBlock(channels[i], channels[i - 1], channels[i - 1]))

        self.head = nn.Sequential(
            ReflectConv2d(channels[0], channels[0], 3, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels[0], out_channels, 1, padding=0, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        input_size = x_masked.shape[-2:]

        skips = []
        x = self.stem(x_masked)
        skips.append(x)

        for down in self.encoder:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)

        for up, skip in zip(self.decoder, reversed(skips[:-1])):
            x = up(x, skip)

        x = self.head(x)

        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        if valid.shape[1] != x.shape[1]:
            valid = valid[:, :1].expand(-1, x.shape[1], -1, -1)

        x = torch.where(valid, x, torch.full_like(x, float("nan")))
        return x


