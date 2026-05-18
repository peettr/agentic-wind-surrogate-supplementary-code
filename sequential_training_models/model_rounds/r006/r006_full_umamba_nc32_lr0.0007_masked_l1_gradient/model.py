import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    g = min(8, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(num_groups=g, num_channels=ch)

class ReflectionConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, bias: bool = False) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=0, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleSSMBlock(nn.Module):
    """Bidirectional selective SSM block over a flattened spatial sequence."""

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.d_state = d_state

        self.proj_in = nn.Linear(dim, dim, bias=False)
        self.proj_gate = nn.Linear(dim, dim, bias=False)
        self.A_log = nn.Parameter(torch.randn(dim, d_state) * 0.5 - 2.0)
        self.B_proj = nn.Linear(dim, d_state, bias=False)
        self.C_proj = nn.Linear(d_state, dim, bias=False)
        self.D = nn.Parameter(torch.ones(dim))
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        nn.init.constant_(self.dt_proj.bias, 0.5)
        self.proj_out = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def _scan_direction(self, x_seq: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x_seq.shape

        B_mat = self.B_proj(x_seq)
        dt = F.softplus(self.dt_proj(x_seq))

        A_pos = F.softplus(A)
        bar_A = torch.exp(-dt.unsqueeze(-1) * A_pos.unsqueeze(0).unsqueeze(0))
        bar_B = dt.unsqueeze(-1) * B_mat.unsqueeze(2).expand(-1, -1, dim, -1)

        h = torch.zeros(batch, dim, self.d_state, device=x_seq.device, dtype=x_seq.dtype)
        c_weight = self.C_proj.weight.to(dtype=x_seq.dtype)
        outputs = []
        for t in range(length):
            h = bar_A[:, t] * h + bar_B[:, t]
            y_t = (h * c_weight.unsqueeze(0)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)

        gate = torch.sigmoid(self.proj_gate(x_norm))
        x_proj = self.proj_in(x_norm) * gate

        A = self.A_log
        y_fwd = self._scan_direction(x_proj, A)
        y_bwd = self._scan_direction(x_proj.flip(dims=[1]), A).flip(dims=[1])
        y = y_fwd + y_bwd

        y = y + self.D.view(1, 1, -1) * x_proj
        y = self.proj_out(y)
        return y + residual


class MambaBlock(nn.Module):
    """Mamba SSM block with explicit 2D spatial flatten/unflatten."""

    def __init__(self, dim: int, d_state: int = 16) -> None:
        super().__init__()
        self.mamba = SimpleSSMBlock(dim, d_state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        y_seq = self.mamba(x_seq)
        return y_seq.transpose(1, 2).reshape(b, c, h, w)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ReflectionConv2d(in_ch, out_ch, 3, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
            ReflectionConv2d(out_ch, out_ch, 3, bias=False),
            _gn(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    dh = skip.size(2) - x.size(2)
    dw = skip.size(3) - x.size(3)
    if dh != 0 or dw != 0:
        left = dw // 2
        right = dw - left
        top = dh // 2
        bottom = dh - top
        x = F.pad(x, [left, right, top, bottom], mode="reflect")
    return torch.cat([x, skip], dim=1)


class umamba(nn.Module):
    """U-Mamba: UNet with a Mamba-style SSM bottleneck."""

    SUPPORTED_DEPTHS = (5, 6, 7)

    def __init__(self, in_channels: int = 1, out_channels: int = 1, n_c: int = 16, depth: int = 7) -> None:
        super().__init__()
        if depth not in self.SUPPORTED_DEPTHS:
            raise ValueError(f"depth must be in {self.SUPPORTED_DEPTHS}, got {depth}")

        self.depth = depth
        self.n_c = n_c

        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()

        ch_in = in_channels
        for k in range(depth):
            ch_out = n_c * 2**k
            self.enc.append(ConvBlock(ch_in, ch_out))
            self.pool.append(nn.MaxPool2d(2))
            ch_in = ch_out

        bottleneck_ch = ch_in
        self.bottleneck = nn.Sequential(
            MambaBlock(bottleneck_ch),
            MambaBlock(bottleneck_ch),
            MambaBlock(bottleneck_ch),
            MambaBlock(bottleneck_ch),
        )

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()

        ch_in = bottleneck_ch
        for k in reversed(range(depth)):
            ch_skip = n_c * 2**k
            self.up.append(nn.ConvTranspose2d(ch_in, ch_skip, 2, stride=2))
            self.dec.append(ConvBlock(ch_skip * 2, ch_skip))
            ch_in = ch_skip

        self.head = nn.Sequential(nn.Conv2d(n_c, out_channels, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

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