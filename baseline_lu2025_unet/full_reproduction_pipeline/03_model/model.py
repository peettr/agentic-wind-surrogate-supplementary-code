"""
7-level UNet (6 maxpool + bottleneck).
Channels: 16→32→64→128→256→512→1024
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetLu7Level(nn.Module):
    def __init__(self, n_c: int = 16) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = ConvBlock(1, n_c)
        self.enc2 = ConvBlock(n_c, 2*n_c)
        self.enc3 = ConvBlock(2*n_c, 4*n_c)
        self.enc4 = ConvBlock(4*n_c, 8*n_c)
        self.enc5 = ConvBlock(8*n_c, 16*n_c)
        self.enc6 = ConvBlock(16*n_c, 32*n_c)
        self.bottleneck = ConvBlock(32*n_c, 64*n_c)

        self.up6 = nn.ConvTranspose2d(64*n_c, 32*n_c, 3, stride=2, padding=1, output_padding=1)
        self.dec6 = ConvBlock(64*n_c, 32*n_c)
        self.up5 = nn.ConvTranspose2d(32*n_c, 16*n_c, 3, stride=2, padding=1, output_padding=1)
        self.dec5 = ConvBlock(32*n_c, 16*n_c)
        self.up4 = nn.ConvTranspose2d(16*n_c, 8*n_c, 3, stride=2, padding=1, output_padding=1)
        self.dec4 = ConvBlock(16*n_c, 8*n_c)
        self.up3 = nn.ConvTranspose2d(8*n_c, 4*n_c, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = ConvBlock(8*n_c, 4*n_c)
        self.up2 = nn.ConvTranspose2d(4*n_c, 2*n_c, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = ConvBlock(4*n_c, 2*n_c)
        self.up1 = nn.ConvTranspose2d(2*n_c, n_c, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = ConvBlock(2*n_c, n_c)

        self.out_conv = nn.Sequential(
            nn.Conv2d(n_c, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e6 = self.enc6(self.pool(e5))
        b = self.bottleneck(self.pool(e6))

        d6 = self.dec6(self._pad_cat(self.up6(b), e6))
        d5 = self.dec5(self._pad_cat(self.up5(d6), e5))
        d4 = self.dec4(self._pad_cat(self.up4(d5), e4))
        d3 = self.dec3(self._pad_cat(self.up3(d4), e3))
        d2 = self.dec2(self._pad_cat(self.up2(d3), e2))
        d1 = self.dec1(self._pad_cat(self.up1(d2), e1))
        return self.out_conv(d1)

    @staticmethod
    def _pad_cat(x, skip):
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return torch.cat([x, skip], dim=1)


if __name__ == '__main__':
    model = UNetLu7Level(n_c=16)
    n = sum(p.numel() for p in model.parameters())
    print(f"Params: {n:,}")
    x = torch.randn(1, 1, 640, 640)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
