"""Generated Auto V5 campaign wrapper for dilated_unet."""
import torch
import torch.nn.functional as F

from shared.models import REGISTRY


class Model(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()
        self.input_adapter = (
            torch.nn.Identity()
            if in_channels == 1
            else torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.backbone = REGISTRY.build("dilated_unet", **kwargs)
        self.output_adapter = (
            torch.nn.Identity()
            if out_channels == 1
            else torch.nn.Conv2d(1, out_channels, kernel_size=1)
        )

    def forward(self, x):
        target_hw = x.shape[-2:]
        x = self.input_adapter(x)
        x = self.backbone(x)
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = self.output_adapter(x)
        return x
