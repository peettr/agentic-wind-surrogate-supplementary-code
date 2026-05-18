"""AttentionMamba â€” Attention gates + Mamba SSM hybrid.

Uses attention gates (from AG-UNet) in the encoder for focused skip connections,
combined with Mamba SSM in the decoder for efficient sequential modeling.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSurrogate


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class AttentionGate(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.W_g = nn.Conv2d(ch, ch, 1, bias=False)
        self.W_x = nn.Conv2d(ch, ch, 1, bias=False)
        self.psi = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=False), nn.Sigmoid())
        self.norm = _gn(ch)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != g.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        q = F.relu(self.norm(self.W_x(x) + self.W_g(g)))
        return x * self.psi(q)


class SimpleSSM(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.A = nn.Parameter(torch.randn(dim) * 0.1)
        self.D = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.silu(self.proj(x) * torch.exp(-torch.exp(self.A)) + self.D * x)


class ConvBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), _gn(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), _gn(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.block(x))


class SSMDecoderBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(ch)
        self.ssm = SimpleSSM(ch)
        self.norm = _gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)
        x_flat = self.ssm(x_flat)
        x = x + x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return self.norm(x)


class AttentionMamba(BaseSurrogate):
    """Attention-gated encoder + SSM decoder for wind field prediction.

    Args:
        n_c: base channel count.
        depth: U-Net depth.
    """

    def __init__(self, n_c: int = 48, depth: int = 4) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, n_c, 4, stride=2, padding=1, bias=False), _gn(n_c), nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(ConvBlock(ch), ConvBlock(ch)))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            self.attn_gates.append(AttentionGate(ch))
            ch *= 2

        self.bottleneck = nn.Sequential(ConvBlock(ch), ConvBlock(ch))

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False), _gn(ch // 2), nn.GELU()))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False), _gn(ch // 2), nn.GELU(),
                SSMDecoderBlock(ch // 2),
            ))
            ch //= 2

        self.output_proj = nn.Sequential(
            nn.Conv2d(n_c, n_c, 3, padding=1, bias=False), _gn(n_c), nn.GELU(),
            nn.Conv2d(n_c, 1, 1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]  # save original resolution
        x = self.input_proj(x)
        skips = []
        for blocks, down in zip(self.enc_blocks, self.down):
            x = blocks(x)
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for k in range(self.depth):
            x = self.up[k](x)
            skip = self.attn_gates[self.depth - 1 - k](skips[self.depth - 1 - k], x)
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh != 0 or dw != 0:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[k](x)
        return F.interpolate(self.output_proj(x), size=(H, W), mode="bilinear", align_corners=False)



