"""U-Mamba â€” UNet encoder-decoder with Mamba (S6) state space model in bottleneck.

Replaces the bottleneck with a bidirectional Mamba SSM for efficient linear-
complexity long-range dependency modeling. Falls back to a simplified SSM
approximation if mamba_ssm is not installed.

Based on: Ma et al., 2024 (U-Mamba)
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


class SimpleSSMBlock(nn.Module):
    """Bidirectional selective SSM block with proper EMA recurrence.

    Implements a genuine four-direction (row LR/RL, col TB/BT) selective
    scan with discretized A-matrix decay.  Each direction runs an
    element-wise EMA recurrence:
        h_t = bar_A * h_{t-1} + bar_B * x_t
        y_t = C @ h_t + D * x_t
    where bar_A = exp(softplus(dt) * softplus(A)), bar_B = softplus(dt) * B(x_t).
    The four directional outputs are summed to produce the final output.
    """

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        # Input projection + gating
        self.proj_in = nn.Linear(dim, dim, bias=False)
        self.proj_gate = nn.Linear(dim, dim, bias=False)
        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(dim, d_state) * 0.5 - 2.0)  # init to small positive after softplus
        self.B_proj = nn.Linear(dim, d_state, bias=False)
        self.C_proj = nn.Linear(d_state, dim, bias=False)
        self.D = nn.Parameter(torch.ones(dim))
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        nn.init.constant_(self.dt_proj.bias, 0.5)  # softplus(0.5) ~ 0.97
        self.proj_out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def _scan_direction(self, x_seq: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Run selective EMA scan along dim 1 (sequence dim).

        Args:
            x_seq: (B, L, dim) input sequence
            A: (dim, d_state) discretized decay matrix (positive)
        Returns:
            (B, L, dim) output sequence
        """
        B_batch, L, dim = x_seq.shape
        d_state = self.d_state

        # Compute input-dependent B and dt
        B_mat = self.B_proj(x_seq)  # (B, L, d_state)
        dt = F.softplus(self.dt_proj(x_seq))  # (B, L, dim)

        # Discretize: bar_A = exp(-dt * A_pos), bar_B = dt * B
        A_pos = F.softplus(A)  # (dim, d_state)
        bar_A = torch.exp(-dt.unsqueeze(-1) * A_pos.unsqueeze(0).unsqueeze(0))  # (B, L, dim, d_state)
        dt_exp = dt.unsqueeze(-1)  # (B, L, dim, 1)
        B_exp = B_mat.unsqueeze(2)  # (B, L, 1, d_state)
        bar_B = dt_exp * B_exp.expand(-1, -1, dim, -1)  # (B, L, dim, d_state)

        # Run recurrence
        h = torch.zeros(B_batch, dim, d_state, device=x_seq.device, dtype=x_seq.dtype)
        outputs = []
        for t in range(L):
            h = bar_A[:, t] * h + bar_B[:, t]  # (B, dim, d_state)
            y_t = self.C_proj(h.reshape(B_batch * dim, d_state)).reshape(B_batch, dim)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, L, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_norm = self.norm(x_flat)

        # Gating
        gate = torch.sigmoid(self.proj_gate(x_norm))  # (B, N, dim)
        x_proj = self.proj_in(x_norm) * gate  # (B, N, dim)

        A = self.A_log  # (dim, d_state)

        # Four-direction scan
        # Row-major: (B, H, W, dim)
        x_2d = x_proj.reshape(B, H, W, C)

        # Row left-to-right
        x_lr = x_2d.reshape(B * H, W, C)
        y_lr = self._scan_direction(x_lr, A).reshape(B, H, W, C)

        # Row right-to-left
        x_rl = x_2d.flip(dims=[2]).reshape(B * H, W, C)
        y_rl = self._scan_direction(x_rl, A).reshape(B, H, W, C).flip(dims=[2])

        # Col top-to-bottom
        x_tb = x_2d.permute(0, 2, 1, 3).reshape(B * W, H, C)
        y_tb = self._scan_direction(x_tb, A).reshape(B, W, H, C).permute(0, 2, 1, 3)

        # Col bottom-to-top
        x_bt = x_2d.flip(dims=[1]).permute(0, 2, 1, 3).reshape(B * W, H, C)
        y_bt = self._scan_direction(x_bt, A).reshape(B, W, H, C).permute(0, 2, 1, 3).flip(dims=[1])

        # Sum four directions
        y = y_lr + y_rl + y_tb + y_bt  # (B, H, W, C)
        y = y.reshape(B, H * W, C)

        # Add skip connection (D term)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_proj

        y = self.proj_out(y)
        y = y + x_flat  # residual
        return y.reshape(B, H, W, C).permute(0, 3, 1, 2)


class MambaBlock(nn.Module):
    """Mamba SSM block with residual connection."""

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        # Force SimpleSSMBlock (mamba_ssm may not be available or compatible)
        self.mamba = SimpleSSMBlock(dim, d_state)
        self.use_real_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_real_mamba:
            B, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            y = self.mamba(x_flat)
            return y.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.mamba(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    dh = skip.size(2) - x.size(2)
    dw = skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
    return torch.cat([x, skip], dim=1)


class UMamba(BaseSurrogate):
    """U-Mamba: UNet with Mamba SSM in the bottleneck for global context.

    Encoder and decoder use standard convolutions; the bottleneck uses
    bidirectional Mamba SSM blocks for efficient long-range modeling.

    Args:
        depth: number of encoder stages (5, 6, or 7).
        n_c: base channel count.
        d_state: SSM state dimension.
        n_ssm_blocks: number of Mamba blocks in bottleneck.
    """

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, depth: int = 7, n_c: int = 32, d_state: int = 16,
                 n_ssm_blocks: int = 4) -> None:
        super().__init__()
        if depth not in self.SUPPORTED_DEPTHS:
            raise ValueError(f"depth must be in {self.SUPPORTED_DEPTHS}, got {depth}")
        self.depth = depth
        self.n_c = n_c

        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        ch_in = 1
        for k in range(depth):
            ch_out = n_c * 2 ** k
            self.enc.append(ConvBlock(ch_in, ch_out))
            self.pool.append(nn.MaxPool2d(2))
            ch_in = ch_out

        bottleneck_ch = ch_in  # = n_c * 2^(depth-1)
        self.bottleneck = nn.Sequential(*[
            MambaBlock(bottleneck_ch, d_state) for _ in range(n_ssm_blocks)
        ])

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        ch_in = bottleneck_ch
        for k in reversed(range(depth)):
            ch_skip = n_c * 2 ** k
            self.up.append(nn.ConvTranspose2d(ch_in, ch_skip, 2, stride=2))
            self.dec.append(ConvBlock(ch_skip * 2, ch_skip))
            ch_in = ch_skip

        self.head = nn.Sequential(nn.Conv2d(n_c, 1, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc_block, pool in zip(self.enc, self.pool):
            x = enc_block(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for k in range(self.depth):
            x = self.up[k](x)
            skip = skips[self.depth - 1 - k]
            x = _pad_cat(x, skip)
            x = self.dec[k](x)
        return self.head(x)



