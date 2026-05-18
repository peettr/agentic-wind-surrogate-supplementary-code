import torch
import torch.nn as nn
import torch.nn.functional as F


class RefConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        pad = kernel_size // 2
        self.pad = nn.ReflectionPad2d(pad) if pad > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(self.pad(x))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class StateSpaceBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, 1)
        self.dw = RefConv2d(channels, channels, 3, groups=channels)
        self.mix_h = nn.Conv2d(channels, channels, 1, groups=channels)
        self.mix_w = nn.Conv2d(channels, channels, 1, groups=channels)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.attn = ChannelAttention(channels)

    def forward(self, x):
        r = x
        u, g = self.in_proj(self.norm(x)).chunk(2, dim=1)
        u = self.dw(u)

        h_lr = torch.cumsum(u, dim=3)
        h_rl = torch.flip(torch.cumsum(torch.flip(u, dims=[3]), dim=3), dims=[3])
        v_tb = torch.cumsum(u, dim=2)
        v_bt = torch.flip(torch.cumsum(torch.flip(u, dims=[2]), dim=2), dims=[2])

        h = self.mix_h((h_lr + h_rl) / max(u.shape[3], 1))
        v = self.mix_w((v_tb + v_bt) / max(u.shape[2], 1))
        y = (u + h + v) * torch.sigmoid(g)
        y = self.out_proj(self.attn(y))
        return r + y


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = RefConv2d(channels, channels, 3)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = RefConv2d(channels, channels, 3)
        self.attn = ChannelAttention(channels)

    def forward(self, x):
        y = self.conv1(F.gelu(self.norm1(x)))
        y = self.conv2(F.gelu(self.norm2(y)))
        y = self.attn(y)
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = RefConv2d(in_channels, out_channels, 3)
        self.block1 = ResidualBlock(out_channels)
        self.block2 = StateSpaceBlock(out_channels)

    def forward(self, x):
        x = self.proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = RefConv2d(in_channels + skip_channels, out_channels, 3)
        self.block1 = ResidualBlock(out_channels)
        self.block2 = StateSpaceBlock(out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.block1(x)
        x = self.block2(x)
        return x


class attentive_state_space_restoration_unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        depth = max(int(depth), 1)
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = RefConv2d(in_channels, channels[0], 3)
        self.encoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(EncoderBlock(channels[i - 1] if i > 0 else channels[0], channels[i]))

        self.down = nn.AvgPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1]),
            StateSpaceBlock(channels[-1]),
            ResidualBlock(channels[-1]),
        )

        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.decoders.append(DecoderBlock(channels[i + 1], channels[i], channels[i]))

        self.head = nn.Sequential(
            RefConv2d(channels[0], channels[0], 3),
            nn.GELU(),
            RefConv2d(channels[0], out_channels, 3),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        h = self.stem(x_masked)
        skips = []
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if i != len(self.encoders) - 1:
                h = self.down(h)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            h = dec(h, skip)

        out = self.head(h)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out_valid = valid
        if out_valid.shape[1] != out.shape[1]:
            if out_valid.shape[1] == 1:
                out_valid = out_valid.expand(-1, out.shape[1], -1, -1)
            else:
                out_valid = out_valid.all(dim=1, keepdim=True)
                if out_valid.shape[1] != out.shape[1]:
                    out_valid = out_valid.expand(-1, out.shape[1], -1, -1)

        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out


