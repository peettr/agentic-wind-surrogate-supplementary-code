"""Generated standalone Auto V5 model for mamba_attention.

This generated file is the training source of truth for this run.
Runtime model construction is local to this file rather than registry delegation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from abc import ABC, abstractmethod


class BaseSurrogate(nn.Module, ABC):
    """Standalone BaseSurrogate copy for generated models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Auto V5 generated training source-of-truth models."""

    def check_shapes(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[1:] != (1, 640, 640):
            raise ValueError(f"Input shape mismatch: expected (B, 1, 640, 640), got {tuple(x.shape)}")
        if y.shape[1:] != (1, 640, 640):
            raise ValueError(f"Output shape mismatch: expected (B, 1, 640, 640), got {tuple(y.shape)}")



def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, ch: int, reduction: int = 4) -> None:
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1, bias=False), nn.GELU(),
            nn.Conv2d(ch // reduction, ch, 1, bias=False),
        )
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        x = x * ca
        sa = self.spatial(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1))
        return x * sa


class SimpleSSM(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.A = nn.Parameter(torch.randn(dim) * 0.1)
        self.D = nn.Parameter(torch.randn(dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.silu(self.proj(x) * torch.exp(-torch.exp(self.A)) + self.D * x)


class MambaAttentionBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.cbam = CBAM(ch)
        self.norm1 = _gn(ch)
        self.norm2 = _gn(ch)
        self.ssm = SimpleSSM(ch)
        self.conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.cbam(self.norm1(x))
        B, C, H, W = x.shape
        x_ssm = x.flatten(2).permute(0, 2, 1)
        x_ssm = self.ssm(x_ssm)
        x = x + x_ssm.permute(0, 2, 1).reshape(B, C, H, W)
        x = x + self.conv(self.norm2(x))
        return x


class MambaAttention(BaseSurrogate):
    """QuadMamba + CBAM attention for wind field prediction.

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
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(MambaAttentionBlock(ch), MambaAttentionBlock(ch)))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            ch *= 2

        self.bottleneck = nn.Sequential(MambaAttentionBlock(ch), MambaAttentionBlock(ch))

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False), _gn(ch // 2), nn.GELU()))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False), _gn(ch // 2), nn.GELU(),
                MambaAttentionBlock(ch // 2),
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
            skip = skips[self.depth - 1 - k]
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh != 0 or dw != 0:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[k](x)
        return F.interpolate(self.output_proj(x), size=(H, W), mode="bilinear", align_corners=False)


class Model(MambaAttention):
    """Training entrypoint for generated Auto V5 runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)
