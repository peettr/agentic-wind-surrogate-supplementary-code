"""unet_sdf_7level.py - 7-level UNet with reflection padding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class unet_sdf_7level(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_c: int = 16,
        depth: int = 7,
    ):
        super().__init__()
        if depth != 7:
            raise ValueError("unet_sdf_7level expects depth=7")

        c = n_c
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.down5 = Down(c * 16, c * 16)
        self.down6 = Down(c * 16, c * 16)

        self.up1 = Up(c * 16, c * 16, c * 16)
        self.up2 = Up(c * 16, c * 16, c * 16)
        self.up3 = Up(c * 16, c * 8, c * 8)
        self.up4 = Up(c * 8, c * 4, c * 4)
        self.up5 = Up(c * 4, c * 2, c * 2)
        self.up6 = Up(c * 2, c, c)

        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(x)
        x = torch.where(mask, x, torch.zeros_like(x))

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)

        return self.outc(x)


class ReflectionConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, bias: bool = False):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ReflectionConv2d(in_ch, out_ch, 3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            ReflectionConv2d(out_ch, out_ch, 3, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


