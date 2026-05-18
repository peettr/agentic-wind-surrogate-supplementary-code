import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(ch: int) -> int:
    return next(g for g in range(min(32, ch), 0, -1) if ch % g == 0)


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(_num_groups(ch), ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_ch, out_ch, 3, bias=False),
            nn.GroupNorm(_num_groups(out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AFNOBlock(nn.Module):
    def __init__(self, channels: int, num_modes: int = 32) -> None:
        super().__init__()
        self.num_modes = int(num_modes)
        self.norm = _gn(channels)

        weight = torch.zeros(channels, self.num_modes, self.num_modes, 2)
        weight[..., 0] = 1.0
        self.weight = nn.Parameter(weight)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        spatial_size = h.shape[-2:]

        h_ft = torch.fft.rfft2(h.float(), norm="ortho")
        out_ft = torch.zeros_like(h_ft)

        mh = min(self.num_modes, h_ft.size(-2))
        mw = min(self.num_modes, h_ft.size(-1))
        weight = torch.view_as_complex(self.weight[:, :mh, :mw].contiguous())

        out_ft[:, :, :mh, :mw] = h_ft[:, :, :mh, :mw] * weight.unsqueeze(0)

        h = torch.fft.irfft2(out_ft, s=spatial_size, norm="ortho").to(dtype=x.dtype)
        return residual + self.scale.to(dtype=x.dtype) * h


class AFNOBottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        num_afno_layers: int = 1,
        num_modes: int = 32,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *(AFNOBlock(channels, num_modes=num_modes) for _ in range(num_afno_layers))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class unet_afno(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_c: int = 16,
        depth: int = 7,
        afno_layers: int = 1,
        afno_modes: int = 32,
        training: dict | None = None,
    ) -> None:
        super().__init__()
        del training

        self.pool = nn.MaxPool2d(2, 2)

        enc_channels = [n_c * (2**k) for k in range(depth)]
        bottleneck_ch = enc_channels[-1] * 2

        self.enc_blocks = nn.ModuleList()
        prev = in_channels
        for ch in enc_channels:
            self.enc_blocks.append(ConvBlock(prev, ch))
            prev = ch

        if afno_layers > 0:
            self.bottleneck = nn.Sequential(
                ConvBlock(enc_channels[-1], bottleneck_ch),
                AFNOBottleneck(
                    bottleneck_ch,
                    num_afno_layers=afno_layers,
                    num_modes=afno_modes,
                ),
            )
        else:
            self.bottleneck = ConvBlock(enc_channels[-1], bottleneck_ch)

        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        prev = bottleneck_ch
        for ch in reversed(enc_channels):
            self.up_blocks.append(UpConv(prev, ch))
            self.dec_blocks.append(ConvBlock(2 * ch, ch))
            prev = ch

        self.out_conv = nn.Sequential(
            nn.Conv2d(n_c, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.masked_fill(torch.isnan(x), 0.0)

        skips: list[torch.Tensor] = []
        h = x
        for i, enc in enumerate(self.enc_blocks):
            h = enc(h if i == 0 else self.pool(h))
            skips.append(h)

        h = self.bottleneck(self.pool(skips[-1]))

        for up, dec, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            h = self._pad_cat(up(h), skip)
            h = dec(h)

        x = self.out_conv(h)
        return x

    @staticmethod
    def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)

        if dh < 0:
            crop = -dh
            top = crop // 2
            bottom = crop - top
            x = x[:, :, top : x.size(2) - bottom, :]

        if dw < 0:
            crop = -dw
            left = crop // 2
            right = crop - left
            x = x[:, :, :, left : x.size(3) - right]

        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)

        if dh > 0 or dw > 0:
            x = F.pad(
                x,
                [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2],
                mode="reflect",
            )

        return torch.cat([x, skip], dim=1)


