"""Generated standalone Grid model for hrnet.

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
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class BasicBlock(nn.Module):
    """Two 3x3 conv + GN + ReLU."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            _gn(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            _gn(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class HRStage(nn.Module):
    """One HRNet stage: parallel streams at different resolutions + cross-resolution fusion."""

    def __init__(self, in_channels: list[int], out_channels: list[int],
                 n_blocks: int = 4) -> None:
        super().__init__()
        self.n_streams = len(in_channels)

        # Stream branches
        self.branches = nn.ModuleList()
        for i in range(self.n_streams):
            layers = nn.Sequential(*[BasicBlock(in_channels[i]) for _ in range(n_blocks)])
            self.branches.append(layers)

        # Fusion: every stream receives info from all other streams
        self.fuse_layers = nn.ModuleList()
        for i in range(self.n_streams):
            fuse = nn.ModuleList()
            for j in range(self.n_streams):
                if j == i:
                    fuse.append(nn.Identity())
                elif j > i:
                    # Downsample stream j to match stream i
                    fuse.append(nn.Sequential(
                        nn.Conv2d(out_channels[j], out_channels[i], 1, bias=False),
                        _gn(out_channels[i]),
                    ))
                else:
                    # Upsample stream j to match stream i
                    fuse.append(nn.Sequential(
                        nn.Conv2d(out_channels[j], out_channels[i], 1, bias=False),
                        _gn(out_channels[i]),
                    ))
            self.fuse_layers.append(fuse)

    def _resize(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if x.shape[2] == target_h and x.shape[3] == target_w:
            return x
        if x.shape[2] > target_h:
            return F.adaptive_avg_pool2d(x, (target_h, target_w))
        return F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        # Apply branch convolutions
        branch_outs = [branch(x) for branch, x in zip(self.branches, inputs)]

        # Fuse across branches
        fused = []
        target_h, target_w = branch_outs[0].shape[2], branch_outs[0].shape[3]
        for i in range(self.n_streams):
            h_i, w_i = branch_outs[i].shape[2], branch_outs[i].shape[3]
            parts = []
            for j in range(self.n_streams):
                x = self.fuse_layers[i][j](branch_outs[j])
                x = self._resize(x, h_i, w_i)
                parts.append(x)
            fused.append(sum(parts) / len(parts))  # Average fusion
        return fused


class HRNetSurrogate(BaseSurrogate):
    """HRNet-based regression network for dense wind field prediction.

    Maintains parallel multi-resolution streams, avoiding the information
    bottleneck of traditional UNet.

    Args:
        depth: number of HR stages (3, 4, or 5).
        n_c: base channel count for highest resolution stream.
        n_blocks: number of basic blocks per stage.
    """

    SUPPORTED_DEPTHS = (3, 4, 5)

    def __init__(self, depth: int = 4, n_c: int = 24, n_blocks: int = 4) -> None:
        super().__init__()
        if depth not in self.SUPPORTED_DEPTHS:
            raise ValueError(f"depth must be in {self.SUPPORTED_DEPTHS}, got {depth}")
        self.depth = depth
        self.n_c = n_c

        # Stem: reduce spatial size by 2
        self.stem = nn.Sequential(
            nn.Conv2d(1, n_c, 3, stride=2, padding=1, bias=False),
            _gn(n_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_c, n_c, 3, stride=2, padding=1, bias=False),
            _gn(n_c),
            nn.ReLU(inplace=True),
        )

        # Stage 1: single stream -> 2 streams
        self.transition1 = nn.Conv2d(n_c, n_c, 1, bias=False)
        self.transition1_down = nn.Sequential(
            nn.Conv2d(n_c, n_c * 2, 3, stride=2, padding=1, bias=False),
            _gn(n_c * 2),
        )
        self.stage1 = HRStage([n_c, n_c * 2], [n_c, n_c * 2], n_blocks)

        # Additional stages: add one more stream per stage
        self.transitions = nn.ModuleList()
        self.stages = nn.ModuleList()
        current_channels = [n_c, n_c * 2]
        for s in range(1, depth):
            new_ch = n_c * 2 ** (s + 1)
            # Transition: add a new lower-res stream
            trans = nn.ModuleList()
            # Downsample the lowest-res stream
            trans.append(nn.Sequential(
                nn.Conv2d(current_channels[-1], new_ch, 3, stride=2, padding=1, bias=False),
                _gn(new_ch),
            ))
            self.transitions.append(trans)
            out_channels = current_channels + [new_ch]
            self.stages.append(HRStage(out_channels, out_channels, n_blocks))
            current_channels = out_channels

        # Output head: upsample all streams to original resolution and concatenate
        total_ch = sum(current_channels)
        self.output_head = nn.Sequential(
            nn.Conv2d(total_ch, n_c, 1, bias=False),
            _gn(n_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_c, 1, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[2], x.shape[3]

        x = self.stem(x)  # (B, n_c, H/4, W/4)

        # Stage 1: create 2 streams
        s1_high = self.transition1(x)
        s1_low = self.transition1_down(x)
        streams = self.stage1([s1_high, s1_low])

        # Additional stages
        for s in range(self.depth - 1):
            new_stream = self.transitions[s][0](streams[-1])
            streams.append(new_stream)
            streams = self.stages[s](streams)

        # Upsample all streams to same resolution and concatenate
        target_h, target_w = streams[0].shape[2], streams[0].shape[3]
        all_features = []
        for s in streams:
            s_up = F.interpolate(s, size=(target_h, target_w), mode='bilinear', align_corners=False)
            all_features.append(s_up)
        x = torch.cat(all_features, dim=1)

        x = self.output_head(x)
        # Upsample to original resolution
        x = F.interpolate(x, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        return x


class Model(HRNetSurrogate):
    """Training entrypoint for generated Grid runs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        kwargs.pop('training', None)
        super().__init__(**kwargs)



