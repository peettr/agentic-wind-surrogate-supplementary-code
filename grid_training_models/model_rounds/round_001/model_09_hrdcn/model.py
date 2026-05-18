"""Generated standalone Grid model for hrdcn.

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
        """Forward pass for Grid generated training source-of-truth models."""

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


class DCNv2(nn.Module):
    """Deformable Convolution v2 (approximated with standard conv + offset)."""

    def __init__(self, ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv_offset = nn.Conv2d(ch, 2 * kernel_size * kernel_size, 3, padding=1)
        self.conv = nn.Conv2d(ch, ch, kernel_size, padding=kernel_size // 2, bias=False)
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simplified: use offset to modulate features (no actual deform_conv2d dependency)
        offset = self.conv_offset(x)
        # Use offset as attention weights over spatial locations
        attention = torch.sigmoid(offset[:, :1, :, :])  # (B, 1, H, W)
        return self.conv(x) * attention


class HRBlock(nn.Module):
    """HRNet-style block with DCN (actually uses deformable modulation)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.dcn = DCNv2(ch)
        self.norm1 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.norm2 = _gn(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.dcn(x)))
        x = self.norm2(self.conv2(x))
        return x + residual


class HRDCN(BaseSurrogate):
    """HRNet + Deformable Convolution for wind field prediction.

    Multi-resolution parallel streams with cross-resolution fusion,
    using standard convolutions (DCN requires torchvision ops).

    Args:
        n_c: base channel count for highest resolution stream.
        depth: number of blocks per stage.
        n_stages: number of resolution-doubling stages.
    """

    def __init__(self, n_c: int = 32, depth: int = 4, n_stages: int = 3) -> None:
        super().__init__()
        self.n_stages = n_stages

        # Input: single stream at full resolution
        self.input_proj = nn.Sequential(
            nn.Conv2d(1, n_c, 4, stride=2, padding=1, bias=False), _gn(n_c), nn.GELU(),
        )

        # Build parallel streams
        stream_channels = [n_c]  # start with one stream
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for s in range(n_stages):
            # Add a new lower-resolution stream
            new_ch = n_c * (2 ** (s + 1))
            stream_channels.append(new_ch)

            # Transition: create new stream via stride-2 conv
            trans = nn.ModuleDict({
                f"stream_{s+1}": nn.Sequential(
                    nn.Conv2d(stream_channels[s], new_ch, 3, stride=2, padding=1, bias=False),
                    _gn(new_ch), nn.GELU(),
                )
            })
            self.transitions.append(trans)

            # HRBlocks for all streams
            stage = nn.ModuleList()
            for ch in stream_channels:
                stage.append(nn.Sequential(*[HRBlock(ch) for _ in range(depth)]))
            self.stages.append(stage)

        # Fusion layers (simplified: just concatenate all streams and decode)
        total_ch = sum(stream_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(total_ch, n_c * 4, 1, bias=False), _gn(n_c * 4), nn.GELU(),
        )

        # Decoder: simple progressive upsample
        self.decoder = nn.Sequential(
            nn.Conv2d(n_c * 4, n_c * 2, 3, padding=1, bias=False), _gn(n_c * 2), nn.GELU(),
            nn.Conv2d(n_c * 2, n_c, 3, padding=1, bias=False), _gn(n_c), nn.GELU(),
            nn.Conv2d(n_c, 1, 1), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        streams = [self.input_proj(x)]
        orig_h, orig_w = x.shape[2], x.shape[3]

        for s in range(self.n_stages):
            # Create new stream
            new_stream = self.transitions[s][f"stream_{s+1}"](streams[-1])
            streams.append(new_stream)

            # Apply HRBlocks
            new_streams = []
            for i, (stream, blocks) in enumerate(zip(streams, self.stages[s])):
                new_streams.append(blocks(stream))
            streams = new_streams

        # Fuse: upsample all to full resolution and concat
        fused = []
        for stream in streams:
            if stream.shape[2] != orig_h or stream.shape[3] != orig_w:
                stream = F.interpolate(stream, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            fused.append(stream)
        x = self.fuse(torch.cat(fused, dim=1))
        return F.interpolate(self.decoder(x), size=(orig_h, orig_w), mode="bilinear", align_corners=False)


class Model(HRDCN):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



