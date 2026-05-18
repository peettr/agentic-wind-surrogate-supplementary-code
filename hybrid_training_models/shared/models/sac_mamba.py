"""SACMamba — Self-Adaptive Conv encoder + SSM bottleneck.

Takes SACUNet's dynamic convolution mechanism for local feature extraction
and replaces the standard bottleneck with UMamba's SSM for long-range
sequence modeling. Best of both worlds: adaptive local + global SSM.
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


class SelfAdaptiveConv(nn.Module):
    """Dynamic 1×1 + 3×3 convolution with input-dependent mixing."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(ch, ch, 1, bias=False)
        self.conv3x3 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, 2), nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x)  # (B, 2)
        return w[:, 0:1, None, None] * self.conv1x1(x) + w[:, 1:2, None, None] * self.conv3x3(x)


class SimpleSSMBlock(nn.Module):
    """Simplified SSM block (no mamba_ssm dependency)."""

    def __init__(self, dim: int, dt_rank: int = 4) -> None:
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * 2)
        self.dt_proj = nn.Linear(dt_rank, dim)
        self.A_log = nn.Parameter(torch.randn(dim) * 0.1)
        self.D = nn.Parameter(torch.randn(dim) * 0.1)
        self.proj_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dt_rank = dt_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        residual = x
        x = self.norm(x)
        xz = self.proj_in(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        A = -torch.exp(self.A_log)
        dt = F.softplus(self.dt_proj(x_ssm[..., :self.dt_rank]))
        y = x_ssm * torch.exp(A * dt) + self.D * x_ssm
        y = y * F.silu(z)

        return residual + self.proj_out(y)


class SACConvBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.sac = SelfAdaptiveConv(ch)
        self.norm = _gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.norm(self.sac(x)))


class SACMamba(BaseSurrogate):
    """Self-Adaptive Conv + SSM bottleneck for wind field prediction.

    Args:
        n_c: base channel count.
        depth: U-Net depth.
        ssm_layers: number of SSM layers in bottleneck.
    """

    def __init__(self, n_c: int = 48, depth: int = 4, ssm_layers: int = 4) -> None:
        super().__init__()
        self.depth = depth

        self.input_proj = nn.Sequential(
            nn.Conv2d(1, n_c, 4, stride=2, padding=1, bias=False), _gn(n_c), nn.GELU(),
        )

        self.enc_blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        ch = n_c
        for _ in range(depth):
            self.enc_blocks.append(nn.Sequential(SACConvBlock(ch), SACConvBlock(ch)))
            self.down.append(nn.Sequential(nn.Conv2d(ch, ch * 2, 2, stride=2, bias=False), _gn(ch * 2), nn.GELU()))
            ch *= 2

        # SSM bottleneck
        self.ssm = nn.Sequential(*[SimpleSSMBlock(ch) for _ in range(ssm_layers)])
        self.ssm_norm = _gn(ch)

        self.up = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.up.append(nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2, bias=False), _gn(ch // 2), nn.GELU()))
            self.dec_blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch // 2, 1, bias=False), _gn(ch // 2), nn.GELU(),
                SACConvBlock(ch // 2),
            ))
            ch //= 2

        self.output_proj = nn.Sequential(
            nn.Conv2d(n_c, n_c, 3, padding=1, bias=False), _gn(n_c), nn.GELU(),
            nn.Conv2d(n_c, 1, 1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_orig, W_orig = x.shape[2], x.shape[3]  # save original resolution
        x = self.input_proj(x)
        skips = []
        for blocks, down in zip(self.enc_blocks, self.down):
            x = blocks(x)
            skips.append(x)
            x = down(x)
        # SSM bottleneck: flatten spatial → sequence → SSM → reshape
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        x_flat = self.ssm(x_flat)
        x = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.ssm_norm(x)

        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh != 0 or dw != 0:
                x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[k](x)
        return F.interpolate(self.output_proj(x), size=(H_orig, W_orig), mode="bilinear", align_corners=False)
