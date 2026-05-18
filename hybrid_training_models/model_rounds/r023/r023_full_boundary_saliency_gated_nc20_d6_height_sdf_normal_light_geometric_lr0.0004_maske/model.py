import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(num_channels, max_groups=8):
    groups = min(max_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ReflectConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, bias=bias),
            _gn(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ReflectConv(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, bias=False),
            _gn(out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class BoundarySaliencyGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.saliency = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=0, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat, source):
        gx = source[:, :, :, 1:] - source[:, :, :, :-1]
        gy = source[:, :, 1:, :] - source[:, :, :-1, :]
        gx = F.pad(gx, (0, 1, 0, 0), mode="reflect")
        gy = F.pad(gy, (0, 0, 0, 1), mode="reflect")
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        edge = edge.mean(dim=1, keepdim=True)

        if edge.shape[-2:] != feat.shape[-2:]:
            edge = F.interpolate(edge, size=feat.shape[-2:], mode="bilinear", align_corners=False)

        gate = self.saliency(torch.cat([feat, edge], dim=1))
        return feat * (1.0 + gate)


class boundary_saliency_gated(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.stem = ResidualBlock(in_channels, channels[0])
        self.encoders = nn.ModuleList()
        self.gates = nn.ModuleList()

        for i in range(1, depth):
            self.encoders.append(ResidualBlock(channels[i - 1], channels[i]))
            self.gates.append(BoundarySaliencyGate(channels[i]))

        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1]),
            ResidualBlock(channels[-1], channels[-1]),
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            self.up_projs.append(nn.Conv2d(channels[i + 1], channels[i], kernel_size=1, padding=0, bias=False))
            self.decoders.append(ResidualBlock(channels[i] * 2, channels[i]))

        self.head = nn.Sequential(
            ReflectConv(channels[0], channels[0]),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=0),
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        h = self.stem(x_masked)
        skips.append(h)

        for encoder, gate in zip(self.encoders, self.gates):
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            h = encoder(h)
            h = gate(h, x_masked)
            skips.append(h)

        h = self.bottleneck(h)

        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips[:-1])):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = up_proj(h)
            h = decoder(torch.cat([h, skip], dim=1))

        output = self.head(h)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        valid_out = valid if valid.shape[1] == output.shape[1] else valid.all(dim=1, keepdim=True)
        output = output.masked_fill(~valid_out, float("nan"))
        return output


