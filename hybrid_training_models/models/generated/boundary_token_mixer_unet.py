import torch
import torch.nn as nn
import torch.nn.functional as F

class boundary_token_mixer_unet(nn.Module):
    class RefConv(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size=3, groups=1, bias=False):
            super().__init__()
            pad = kernel_size // 2
            self.pad = nn.ReflectionPad2d(pad)
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, groups=groups, bias=bias)

        def forward(self, x):
            return self.conv(self.pad(x))

    class Block(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.conv1 = boundary_token_mixer_unet.RefConv(in_ch, out_ch, 3, bias=False)
            self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.dw = boundary_token_mixer_unet.RefConv(out_ch, out_ch, 3, groups=out_ch, bias=False)
            self.pw = nn.Conv2d(out_ch, out_ch, 1, bias=False)
            self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            r = self.proj(x)
            x = self.act(self.norm1(self.conv1(x)))
            x = self.norm2(self.pw(self.dw(x)))
            return self.act(x + r)

    class BoundaryMixer(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.edge_proj = nn.Sequential(
                nn.Linear(ch * 4, ch),
                nn.SiLU(inplace=True),
                nn.Linear(ch, ch)
            )
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, ch, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            top = x[:, :, 0, :].mean(dim=-1)
            bottom = x[:, :, -1, :].mean(dim=-1)
            left = x[:, :, :, 0].mean(dim=-1)
            right = x[:, :, :, -1].mean(dim=-1)
            token = torch.cat([top, bottom, left, right], dim=1)
            token = self.edge_proj(token).unsqueeze(-1).unsqueeze(-1)
            return x + token * self.gate(x)

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        depth = max(1, int(depth))
        channels = [min(n_c * (2 ** i), n_c * 8) for i in range(depth)]

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        prev = in_channels
        for ch in channels:
            self.encoders.append(nn.Sequential(
                self.Block(prev, ch),
                self.Block(ch, ch),
                self.BoundaryMixer(ch)
            ))
            self.downs.append(nn.AvgPool2d(2))
            prev = ch

        self.bottleneck = nn.Sequential(
            self.Block(channels[-1], channels[-1]),
            self.BoundaryMixer(channels[-1]),
            self.Block(channels[-1], channels[-1])
        )

        self.up_projs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        dec_in = channels[-1]
        for ch in reversed(channels):
            self.up_projs.append(nn.Conv2d(dec_in, ch, 1, bias=False))
            self.decoders.append(nn.Sequential(
                self.Block(ch * 2, ch),
                self.Block(ch, ch),
                self.BoundaryMixer(ch)
            ))
            dec_in = ch

        self.head = nn.Sequential(
            self.Block(channels[0], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1)
        )

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(valid, x, torch.zeros_like(x))

        skips = []
        y = x_masked

        for enc, down in zip(self.encoders, self.downs):
            y = enc(y)
            skips.append(y)
            y = down(y)

        y = self.bottleneck(y)

        for up_proj, dec, skip in zip(self.up_projs, self.decoders, reversed(skips)):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = up_proj(y)
            y = dec(torch.cat([y, skip], dim=1))

        output = self.head(y)

        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)

        if valid.shape[1] != output.shape[1]:
            valid = valid[:, :1].expand(-1, output.shape[1], -1, -1)

        output = output.clone()
        output[~valid] = float("nan")
        return output


