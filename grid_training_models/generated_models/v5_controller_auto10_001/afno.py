"""Generated Auto V5 campaign wrapper for afno."""
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

        # Limit AFNO spectral mixing at 640x640. The backbone default
        # max_modes=0 processes every Fourier mode and OOMs at batch=8 on L40S.
        # repair1 used max_modes=32 and still consumed 66.66 GB on H100;
        # repair2 used max_modes=16 and still OOMed on L40S before epoch 1.
        # repair3 also reduces the AFNO channel width and depth so the activation
        # footprint can fit the L40S smoke tier at batch=8.
        # Callers can still override these values through kwargs for controlled runs.
        afno_kwargs = {"width": 32, "depth": 3, "max_modes": 8, **kwargs}
        self.backbone = REGISTRY.build("afno", **afno_kwargs)
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
