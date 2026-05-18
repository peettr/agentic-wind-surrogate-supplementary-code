"""unet_sdf_7level.py — 7-level UNet with SDF-augmented 3-channel input.

Architecture: identical depth/width to the auto_v2 baseline (7 levels, ~125M params)
but accepts 3 input channels (building_height + SDF + boundary_normal_angle)
instead of the standard 1 channel.

This tests the hypothesis that explicit geometry cues improve wind field prediction.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
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


class UNetSDF(nn.Module):
    """7-level UNet for SDF-augmented 3-channel input.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default 3 (height + SDF + normal_angle).
        Can be set to 1 for backward compatibility.
    base_ch : int
        Base channel width. Default 64 (~125M params total).
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        c = base_ch
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

        self.outc = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# Alias for Auto V6 script_path loading by registry arch_name.
unet_sdf_7level = UNetSDF

# Wrapper for Auto V6 script_path loading by registry arch_name.
class unet_sdf_7level(UNetSDF):
    def __init__(self, in_channels=1, out_channels=1, n_c=16, depth=7, **kwargs):
        import inspect
        sig = inspect.signature(UNetSDF.__init__)
        call_kwargs = {}
        if "in_channels" in sig.parameters:
            call_kwargs["in_channels"] = in_channels
        if "out_channels" in sig.parameters:
            call_kwargs["out_channels"] = out_channels
        if "n_c" in sig.parameters:
            call_kwargs["n_c"] = n_c
        if "base_ch" in sig.parameters:
            call_kwargs["base_ch"] = n_c
        if "depth" in sig.parameters:
            call_kwargs["depth"] = depth
        for _k, _v in kwargs.items():
            if _k in sig.parameters:
                call_kwargs[_k] = _v
        super().__init__(**call_kwargs)
