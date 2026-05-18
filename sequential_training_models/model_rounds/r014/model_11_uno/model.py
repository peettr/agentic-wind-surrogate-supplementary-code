"""U-NO - U-Shaped Neural Operator for 2D dense regression."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class SpectralConv2d(nn.Module):
    """2D Fourier layer."""

    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_ch * out_ch)
        self.weight1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.weight2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            b,
            self.weight1.shape[1],
            h,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )

        m1 = min(self.modes1, h)
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :m1, :m2],
            self.weight1[:, :, :m1, :m2],
        )
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -m1:, :m2],
            self.weight2[:, :, :m1, :m2],
        )
        return torch.fft.irfft2(out_ft, s=(h, w))


class UNOLayer(nn.Module):
    """U-NO layer: spectral conv + pointwise conv + residual."""

    def __init__(self, in_ch: int, out_ch: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(in_ch, out_ch, modes, modes)
        self.local = nn.Conv2d(in_ch, out_ch, 1)
        self.norm = _gn(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.gelu(self.spectral(x) + self.local(x)))


class uno(nn.Module):
    """U-Shaped Neural Operator for dense wind field prediction."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, n_c: int = 16, depth: int = 7) -> None:
        super().__init__()
        width = n_c
        modes = 16
        n_layers = 2
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, width, 3, padding=0, bias=False),
            _gn(width),
            nn.GELU(),
        )

        self.enc_layers = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = width
        for _ in range(depth):
            layers = []
            c_in = ch
            for _ in range(n_layers):
                layers.append(UNOLayer(c_in, ch, modes))
                c_in = ch
            self.enc_layers.append(nn.Sequential(*layers))
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False),
                    _gn(ch * 2),
                    nn.GELU(),
                )
            )
            ch *= 2
            modes = max(modes // 2, 4)

        self.bottleneck = nn.Sequential(
            UNOLayer(ch, ch, modes),
            UNOLayer(ch, ch, modes),
        )

        self.up = nn.ModuleList()
        self.dec_layers = nn.ModuleList()
        for _ in range(depth):
            self.up.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False),
                    _gn(ch // 2),
                    nn.GELU(),
                )
            )
            ch //= 2
            layers = []
            c_in = ch * 2
            for i in range(n_layers):
                layers.append(UNOLayer(c_in if i == 0 else ch, ch, modes))
                c_in = ch
            self.dec_layers.append(nn.Sequential(*layers))
            modes = min(modes * 2, 16)

        self.output_proj = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(width, width, 3, padding=0, bias=False),
            _gn(width),
            nn.GELU(),
            nn.Conv2d(width, out_channels, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(x)
        x = torch.where(nan_mask, torch.zeros_like(x), x)

        x = self.input_proj(x)
        skips = []
        for layers, down in zip(self.enc_layers, self.down):
            x = layers(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh != 0 or dw != 0:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2], mode="reflect")
            x = torch.cat([x, skip], dim=1)
            x = self.dec_layers[k](x)

        return self.output_proj(x)


