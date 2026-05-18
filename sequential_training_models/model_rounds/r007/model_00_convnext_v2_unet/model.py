import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_groups(channels: int, max_groups: int = 32) -> int:
    """Return the largest GroupNorm group count <= max_groups that divides channels."""
    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= max_groups and channels % groups == 0:
            return groups
    return 1


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, layer_scale_init: float = 1e-6, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.pad_size = pad
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=0, groups=dim)
        self.norm = nn.GroupNorm(_valid_groups(dim), dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.grn_beta = nn.Parameter(torch.zeros(dim))
        self.grn_gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.pad_size > 0:
            h, w = x.shape[-2:]
            mode = "reflect" if h > self.pad_size and w > self.pad_size else "replicate"
            x = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode=mode)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        gx = torch.norm(x, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        x = self.grn_gamma * x * nx + self.grn_beta + x
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvNeXtV2Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvNeXtV2Block(in_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if x.shape[-2] < 2 or x.shape[-1] < 2:
            return self.proj(x)
        return self.down(x)


class ConvNeXtV2Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvNeXtV2Block(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.up(x))


class convnext_v2_unet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_c: int = 16,
        depth: int = 7,
        kernel_size: int = 7,
        training: dict | None = None,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, n_c, kernel_size=4, stride=4, padding=0),
            nn.GroupNorm(_valid_groups(n_c), n_c),
        )

        self.encoders = nn.ModuleList()
        channels = [n_c]
        for _ in range(depth):
            in_ch = channels[-1]
            out_ch = in_ch * 2
            self.encoders.append(ConvNeXtV2Down(in_ch, out_ch))
            channels.append(out_ch)

        self.bottleneck = nn.Sequential(
            ConvNeXtV2Block(channels[-1], kernel_size=kernel_size),
            ConvNeXtV2Block(channels[-1], kernel_size=kernel_size),
        )

        self.decoders = nn.ModuleList()
        for _ in range(depth):
            in_ch = channels[-1]
            out_ch = in_ch // 2
            self.decoders.append(ConvNeXtV2Up(in_ch + out_ch, out_ch))
            channels.append(out_ch)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(channels[-1], n_c, kernel_size=4, stride=4, padding=0),
            nn.Conv2d(n_c, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.channels = channels
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            x = torch.where(nan_mask, torch.zeros_like(x), x)

        x = self.stem(x)

        skips = [x]
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips[:-1])):
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.head(x)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    for n_c in [16, 24, 32]:
        for depth in [4, 5]:
            model = convnext_v2_unet(n_c=n_c, depth=depth)
            n_params = sum(p.numel() for p in model.parameters())
            inp = torch.randn(2, 1, 640, 640)
            with torch.no_grad():
                out = model(inp)
            print(
                f"convnext_v2_unet n_c={n_c} depth={depth}: "
                f"params={n_params:,} ({n_params / 1e6:.1f}M) out={tuple(out.shape)}"
            )


