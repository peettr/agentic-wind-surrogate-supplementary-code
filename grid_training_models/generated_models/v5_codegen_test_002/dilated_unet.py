"""Auto V5 wrapper for the shared dilated UNet dense regression model."""

from __future__ import annotations

import torch

from shared.models.dilated_unet import DilatedUNet

__all__ = ["Model"]


class Model(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        backbone_kwargs = {
            "depth": kwargs.pop("depth", 7),
            "n_c": kwargs.pop("n_c", 16),
            "dilation": kwargs.pop("dilation", 2),
            "spectral_modes": kwargs.pop("spectral_modes", 0),
        }

        self.input_adapter = (
            torch.nn.Identity()
            if in_channels == 1
            else torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.backbone = DilatedUNet(**backbone_kwargs)
        self.output_adapter = (
            torch.nn.Identity()
            if out_channels == 1
            else torch.nn.Conv2d(1, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_adapter(x)
        x = self.backbone(x)
        x = self.output_adapter(x)
        return x
