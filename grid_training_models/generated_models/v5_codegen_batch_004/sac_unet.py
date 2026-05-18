import torch

from shared.models.sac_unet import SACUNet


class Model(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()
        self.input_adapter = (
            torch.nn.Identity()
            if in_channels == 1
            else torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.backbone = SACUNet(
            depth=kwargs.get('depth', 7),
            n_c=kwargs.get('n_c', 12)
        )
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
