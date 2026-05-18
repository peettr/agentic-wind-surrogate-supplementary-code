import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(num_channels, max_groups=8):
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class gradient_boundary_token_alltoall_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
            super().__init__()
            self.pad = nn.ReflectionPad2d(kernel_size // 2)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(
                gradient_boundary_token_alltoall_unet.ReflectConv(in_channels, out_channels),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
                gradient_boundary_token_alltoall_unet.ReflectConv(out_channels, out_channels),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class BoundaryTokenBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(channels * 4, channels),
                nn.SiLU(inplace=True),
                nn.Linear(channels, channels),
            )
            self.gate = nn.Sequential(
                nn.Linear(channels, channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            top = x[:, :, 0, :].mean(dim=-1)
            bottom = x[:, :, -1, :].mean(dim=-1)
            left = x[:, :, :, 0].mean(dim=-1)
            right = x[:, :, :, -1].mean(dim=-1)
            token = self.proj(torch.cat([top, bottom, left, right], dim=1))
            gate = self.gate(token).unsqueeze(-1).unsqueeze(-1)
            return x + x * gate

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_c = n_c
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]
        self.channels = channels

        self.grad_x = self.ReflectConv(in_channels, in_channels, kernel_size=3, bias=False)
        self.grad_y = self.ReflectConv(in_channels, in_channels, kernel_size=3, bias=False)

        with torch.no_grad():
            kx = torch.zeros(in_channels, in_channels, 3, 3)
            ky = torch.zeros(in_channels, in_channels, 3, 3)
            for i in range(in_channels):
                kx[i, i, 1, 0] = -0.5
                kx[i, i, 1, 2] = 0.5
                ky[i, i, 0, 1] = -0.5
                ky[i, i, 2, 1] = 0.5
            self.grad_x.conv.weight.copy_(kx)
            self.grad_y.conv.weight.copy_(ky)
        self.grad_x.conv.weight.requires_grad_(False)
        self.grad_y.conv.weight.requires_grad_(False)

        enc_in = in_channels * 3
        self.encoders = nn.ModuleList()
        self.boundary_blocks = nn.ModuleList()
        for ch in channels:
            self.encoders.append(self.ConvBlock(enc_in, ch))
            self.boundary_blocks.append(self.BoundaryTokenBlock(ch))
            enc_in = ch

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.ConvBlock(channels[-1], channels[-1])

        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = channels[-1] if i == depth - 1 else channels[i + 1]
            out_ch = channels[i]
            self.up_convs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            self.decoders.append(self.ConvBlock(out_ch + channels[i], out_ch))

        self.final = nn.Sequential(
            self.ReflectConv(channels[0], channels[0]),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.SiLU(inplace=True),
            self.ReflectConv(channels[0], out_channels, kernel_size=1, bias=True),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        gx = self.grad_x(x_masked)
        gy = self.grad_y(x_masked)
        h = torch.cat([x_masked, gx, gy], dim=1)

        skips = []
        for enc, boundary in zip(self.encoders, self.boundary_blocks):
            h = boundary(enc(h))
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.up_convs, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))

        output = self.final(h)
        output = torch.where(valid, output, torch.full_like(output, float("nan")))
        return output


