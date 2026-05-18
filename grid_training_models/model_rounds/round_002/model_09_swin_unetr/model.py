"""Generated standalone Grid model for swin_unetr.

This generated file is the training source of truth for this run.
Runtime model construction is local to this file rather than registry delegation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from abc import ABC, abstractmethod


class BaseSurrogate(nn.Module, ABC):
    """Standalone BaseSurrogate copy for generated models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Grid generated training source-of-truth models."""

    def check_shapes(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[1:] != (1, 640, 640):
            raise ValueError(f"Input shape mismatch: expected (B, 1, 640, 640), got {tuple(x.shape)}")
        if y.shape[1:] != (1, 640, 640):
            raise ValueError(f"Output shape mismatch: expected (B, 1, 640, 640), got {tuple(y.shape)}")



def _gn(ch: int) -> nn.GroupNorm:
    """GroupNorm with safe num_groups."""
    g = min(32, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


def _pad_to_multiple(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad spatial dims to be divisible by window_size. Returns padded tensor and pad amounts."""
    _, _, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [0, pad_w, 0, pad_h])
    return x, (pad_h, pad_w)


class WindowAttention(nn.Module):
    """Multi-head self-attention within a local window."""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
        coords_flatten = coords.flatten(1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (num_windows*B, N, C)"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block: W-MSA or SW-MSA."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8, shift: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = _gn(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = _gn(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        residual = x

        # GroupNorm expects (B, C, H, W), then reshape to (B, H, W, C) for attention
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Pad to window_size multiple
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            # x is (B, H, W, C); F.pad pads from last dim: [C_l,C_r, W_l,W_r, H_l,H_r]
            x = F.pad(x, [0, 0, 0, pad_r, 0, pad_b])
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        nH = Hp // self.window_size
        nW = Wp // self.window_size
        x_windows = shifted_x.view(
            B, nH, self.window_size, nW, self.window_size, C
        ).permute(0, 1, 3, 2, 4, 5).contiguous().view(B * nH * nW, self.window_size * self.window_size, C)

        # Attention
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(B, nH, nW, self.window_size, self.window_size, C)
        shifted_x = attn_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_out = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_out = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x_out = x_out[:, :H, :W, :].contiguous()

        x_out = x_out.permute(0, 3, 1, 2)  # back to (B, C, H, W)
        x = residual + x_out

        # MLP
        x2 = self.norm2(x)
        x2 = x2.permute(0, 2, 3, 1)  # (B, H, W, C)
        x2 = self.mlp(x2)
        x2 = x2.permute(0, 3, 1, 2)  # back to (B, C, H, W)
        x = x + x2
        return x


class PatchMerging(nn.Module):
    """Downsample by merging 2Ã—2 patches (4Ã— reduction in tokens, 2Ã— in channels)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = _gn(dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        # Pad if odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, [0, 0, 0, pad_w, 0, pad_h])
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = self.reduction(x)  # (B, H/2, W/2, 2C)
        return x.permute(0, 3, 1, 2)  # (B, 2C, H/2, W/2)


class ConvDecoderBlock(nn.Module):
    """Standard conv upsample block for UNet decoder."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinUNETR(BaseSurrogate):
    """Swin Transformer encoder + UNet decoder.

    Args:
        in_channels: input channels (1=height, 2=+sdf, 3=+sdf+normal).
        embed_dim: base embedding dimension (48=light, 96=Swin-T).
        window_size: attention window size.
        num_layers: list of Swin blocks per stage (len=4).
        num_heads: list of attention heads per stage (len=4).
        training: dict of training extras â€” ignored.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 48,
        window_size: int = 8,
        num_layers: tuple[int, ...] = (2, 2, 2, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        training: dict | None = None,
        **_extra,
    ) -> None:
        super().__init__()

        dims = [embed_dim * (2 ** i) for i in range(4)]  # [48, 96, 192, 384] for Swin-T

        # Patch embedding (stem): conv 4Ã—4 stride 4
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            _gn(dims[0]),
        )

        # Encoder stages
        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for i in range(4):
            blocks = nn.ModuleList()
            for j in range(num_layers[i]):
                blocks.append(SwinTransformerBlock(
                    dims[i], num_heads[i], window_size, shift=(j % 2 == 1)
                ))
            self.stages.append(nn.Sequential(*blocks))
            if i < 3:
                self.downsample.append(PatchMerging(dims[i]))

        # Decoder (mirrors encoder)
        self.decoders = nn.ModuleList()
        for i in range(3, 0, -1):
            self.decoders.append(ConvDecoderBlock(dims[i], dims[i - 1]))

        # Head
        self.head = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=4),
            nn.Conv2d(dims[0], 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem: 640 â†’ 160
        h = self.patch_embed(x)

        # Encoder
        skips = []
        for i in range(4):
            h = self.stages[i](h)
            skips.append(h)
            if i < 3:
                h = self.downsample[i](h)

        # Decoder
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 2)]  # -2, -3, -4
            h = dec(h, skip)

        # Head: 160 â†’ 640
        out = self.head(h)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out


if __name__ == "__main__":
    configs = [
        (48, (2, 2, 2, 2), (3, 6, 12, 24)),   # light
        (96, (2, 2, 6, 2), (3, 6, 12, 24)),    # Swin-T
    ]
    for embed_dim, nl, nh in configs:
        for ws in [8, 16]:
            m = SwinUNETR(embed_dim=embed_dim, num_layers=nl, num_heads=nh, window_size=ws)
            n = sum(p.numel() for p in m.parameters())
            x = torch.randn(2, 1, 640, 640)
            with torch.no_grad():
                y = m(x)
            print(f"SwinUNETR embed={embed_dim} ws={ws} layers={nl}: params={n:,} ({n/1e6:.1f}M) out={tuple(y.shape)} min={y.min():.4f}")


class Model(SwinUNETR):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



