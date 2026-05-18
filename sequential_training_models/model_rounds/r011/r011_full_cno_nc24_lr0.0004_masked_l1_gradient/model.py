import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while g > 1 and ch % g != 0:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class CNOBlock(nn.Module):
    def __init__(self, ch: int, lift_mult: int = 2) -> None:
        super().__init__()
        lifted = ch * lift_mult
        self.block = nn.Sequential(
            nn.Conv2d(ch, lifted, 3, padding=1, padding_mode="reflect", bias=False),
            _gn(lifted),
            nn.GELU(),
            nn.Conv2d(lifted, lifted, 3, padding=1, padding_mode="reflect", bias=False),
            _gn(lifted),
            nn.GELU(),
            nn.Conv2d(lifted, ch, 3, padding=1, padding_mode="reflect", bias=False),
            _gn(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] <= 1 or x.shape[-1] <= 1:
            return x
        return x + self.block(x)


class cno(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_c: int = 16,
        depth: int = 7,
    ) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, n_c, 4, stride=2, padding=1, padding_mode="reflect", bias=False),
            _gn(n_c),
            nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(CNOBlock(ch), CNOBlock(ch)))
            self.down.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False),
                    _gn(ch * 2),
                    nn.GELU(),
                )
            )
            ch *= 2

        self.bottleneck = nn.Sequential(CNOBlock(ch), CNOBlock(ch))

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False),
                    _gn(ch // 2),
                    nn.GELU(),
                )
            )
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch // 2, 1, bias=False),
                    _gn(ch // 2),
                    nn.GELU(),
                    CNOBlock(ch // 2),
                    CNOBlock(ch // 2),
                )
            )
            ch //= 2

        self.output_proj = nn.Sequential(
            nn.Conv2d(n_c, n_c, 3, padding=1, padding_mode="reflect", bias=False),
            _gn(n_c),
            nn.GELU(),
            nn.Conv2d(n_c, out_channels, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(x)
        x = torch.where(mask, x, torch.zeros_like(x))

        h, w = x.shape[2], x.shape[3]
        x = self.input_proj(x)

        skips = []
        for blocks, down in zip(self.enc_blocks, self.down):
            x = blocks(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck(x)

        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]

            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh != 0 or dw != 0:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2], mode="reflect")

            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[k](x)

        x = self.output_proj(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x