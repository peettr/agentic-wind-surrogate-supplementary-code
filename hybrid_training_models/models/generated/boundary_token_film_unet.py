import torch
import torch.nn as nn
import torch.nn.functional as F

class boundary_token_film_unet(nn.Module):
    class ReflectConv(nn.Module):
        def __init__(self, in_ch, out_ch, k=3):
            super().__init__()
            self.pad = nn.ReflectionPad2d(k // 2)
            self.conv = nn.Conv2d(in_ch, out_ch, k, padding=0, bias=False)

        def forward(self, x):
            return self.conv(self.pad(x))

    class FiLMBlock(nn.Module):
        def __init__(self, in_ch, out_ch, token_ch):
            super().__init__()
            self.conv1 = boundary_token_film_unet.ReflectConv(in_ch, out_ch)
            self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.conv2 = boundary_token_film_unet.ReflectConv(out_ch, out_ch)
            self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
            self.film = nn.Linear(token_ch, out_ch * 2)

        def forward(self, x, token):
            h = F.silu(self.norm1(self.conv1(x)))
            h = self.norm2(self.conv2(h))
            gamma, beta = self.film(token).chunk(2, dim=1)
            h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
            return F.silu(h + self.skip(x))

    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        max_ch = n_c * 8
        chs = [min(n_c * (2 ** i), max_ch) for i in range(depth)]
        token_ch = chs[-1]

        self.token_proj = nn.Sequential(
            nn.Linear(in_channels * 4, token_ch),
            nn.SiLU(),
            nn.Linear(token_ch, token_ch),
        )

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in chs:
            self.encoders.append(self.FiLMBlock(prev, ch, token_ch))
            prev = ch

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self.FiLMBlock(chs[-1], chs[-1], token_ch)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for skip_ch in reversed(chs):
            self.upconvs.append(nn.ConvTranspose2d(prev, skip_ch, kernel_size=2, stride=2))
            self.decoders.append(self.FiLMBlock(skip_ch + skip_ch, skip_ch, token_ch))
            prev = skip_ch

        self.out_pad = nn.ReflectionPad2d(1)
        self.out_conv = nn.Conv2d(chs[0], out_channels, kernel_size=3, padding=0)

    def forward(self, x):
        valid = ~torch.isnan(x)
        x_masked = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        top = x_masked[:, :, 0, :].mean(dim=2)
        bottom = x_masked[:, :, -1, :].mean(dim=2)
        left = x_masked[:, :, :, 0].mean(dim=2)
        right = x_masked[:, :, :, -1].mean(dim=2)
        token = self.token_proj(torch.cat([top, bottom, left, right], dim=1))

        skips = []
        h = x_masked
        for enc in self.encoders:
            h = enc(h, token)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h, token)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = dec(torch.cat([h, skip], dim=1), token)

        out = self.out_conv(self.out_pad(h))
        out_valid = valid if valid.shape[1] == out.shape[1] else valid.all(dim=1, keepdim=True).expand_as(out)
        out = torch.where(out_valid, out, torch.full_like(out, float("nan")))
        return out