"""Compact Auto V5 wrapper for the shared UNet v2 baseline."""

from __future__ import annotations

import torch

from shared.models.unet_v2_baseline import UNetV2Baseline

__all__ = ["Model"]


class Model(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()
        self.model = UNetV2Baseline(in_channels=in_channels, **kwargs)
        self.adapter = (
            torch.nn.Identity()
            if out_channels == 1
            else torch.nn.Conv2d(1, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.adapter(self.model(x))
