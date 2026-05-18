import torch

from shared.models.residual_spectral import ResidualSpectralNet


class Model(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()
        self.input_adapter = (
            torch.nn.Identity()
            if in_channels == 1
            else torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.backbone = ResidualSpectralNet(
            n_c=kwargs.get('n_c', 32),
            depth=kwargs.get('depth', 4),
            modes=kwargs.get('modes', 8)
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
