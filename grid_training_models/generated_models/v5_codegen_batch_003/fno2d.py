import torch

from shared.models.fno2d import FNO2d


class Model(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()

        self.input_adapter = (
            torch.nn.Identity()
            if in_channels == 1
            else torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.backbone = FNO2d(
            width=kwargs.get("width", 32),
            modes=kwargs.get("modes", 12),
            depth=kwargs.get("depth", 4),
            n_blocks=kwargs.get("n_blocks", 1),
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
