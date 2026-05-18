"""Masked loss functions for the UrbanTALES surrogate.

Every loss is NaN-safe. The target tensor ``Y`` contains NaN on building interior
pixels; the input ``X`` encodes buildings as positive values. Valid pixels are
those where ``(~isnan(Y)) & (X <= 0)``.

Appendix A.1 #12 (LOCKED): use ``torch.where(mask, op, zeros)`` â€” never
``mask * op`` â€” because ``0 * NaN = NaN`` in PyTorch, which silently corrupts
gradients.

Every loss signature is ``forward(pred, target, x)`` so that the mask can be
derived from ``x`` (the only rank at which buildings are encoded).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def building_mask(target: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return valid-pixel mask ``(~isnan(target)) & (x_height <= 0)``.
    
    Uses only the first channel of x (building height) for masking,
    regardless of whether SDF/normal channels are present.
    """
    x_height = x[:, 0:1, :, :]  # always use height channel
    return (~torch.isnan(target)) & (x_height <= 0)


def _masked_reduce(per_pixel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum ``per_pixel`` on ``mask``-true pixels, divide by valid count."""
    zeros = torch.zeros_like(per_pixel)
    return torch.where(mask, per_pixel, zeros).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------------------
# Masked L1
# ---------------------------------------------------------------------------
class MaskedL1Loss(nn.Module):
    """Masked L1 loss â€” identical to auto_v2's inline masked_l1()."""

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, x: torch.Tensor,
    ) -> torch.Tensor:
        mask = building_mask(target, x)
        return torch.where(mask, (pred - target).abs(),
                           torch.zeros_like(pred)).sum() / mask.sum()


# ---------------------------------------------------------------------------
# Masked L1 + gradient penalty (Sobel)
# ---------------------------------------------------------------------------
def _sobel_grad(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Gx, Gy)`` Sobel gradients for a ``(B, C, H, W)`` tensor."""
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=img.dtype, device=img.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=img.dtype, device=img.device,
    ).view(1, 1, 3, 3)
    c = img.size(1)
    kx = kx.expand(c, 1, 3, 3)
    ky = ky.expand(c, 1, 3, 3)
    gx = F.conv2d(img, kx, padding=1, groups=c)
    gy = F.conv2d(img, ky, padding=1, groups=c)
    return gx, gy


class MaskedL1GradientLoss(nn.Module):
    """Masked L1 + ``grad_weight`` * Sobel gradient L1.

    Encourages smooth spatial derivatives â€” useful where abrupt jumps in the
    predicted wind field would be unphysical.
    """

    def __init__(self, grad_weight: float = 0.1) -> None:
        super().__init__()
        self.grad_weight = grad_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, x: torch.Tensor,
    ) -> torch.Tensor:
        mask = building_mask(target, x)
        safe_target = torch.where(mask, target, torch.zeros_like(target))
        pointwise = _masked_reduce((pred - safe_target).abs(), mask)

        gx_p, gy_p = _sobel_grad(pred)
        gx_t, gy_t = _sobel_grad(safe_target)
        grad_diff = (gx_p - gx_t).abs() + (gy_p - gy_t).abs()
        return pointwise + self.grad_weight * _masked_reduce(grad_diff, mask)


# ---------------------------------------------------------------------------
# Masked Huber
# ---------------------------------------------------------------------------
class MaskedHuberLoss(nn.Module):
    """Masked Huber (smooth-L1) loss with ``delta`` elbow (default 0.05)."""

    def __init__(self, delta: float = 0.05) -> None:
        super().__init__()
        self.delta = delta

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, x: torch.Tensor,
    ) -> torch.Tensor:
        mask = building_mask(target, x)
        safe_target = torch.where(mask, target, torch.zeros_like(target))
        diff = pred - safe_target
        abs_diff = diff.abs()
        quadratic = 0.5 * diff * diff / self.delta
        linear = abs_diff - 0.5 * self.delta
        per_pixel = torch.where(abs_diff <= self.delta, quadratic, linear)
        return _masked_reduce(per_pixel, mask)


# ---------------------------------------------------------------------------
# Loss library
# ---------------------------------------------------------------------------
class LossLibrary:
    """name â†’ loss-constructor registry."""

    def __init__(self) -> None:
        self._entries: dict[str, Callable[..., nn.Module]] = {}

    def register(self, name: str, ctor: Callable[..., nn.Module]) -> None:
        if name in self._entries:
            raise KeyError(f"Loss '{name}' already registered")
        self._entries[name] = ctor

    def get(self, name: str) -> Callable[..., nn.Module]:
        if name not in self._entries:
            raise KeyError(f"Unknown loss '{name}'. Available: {self.list_all()}")
        return self._entries[name]

    def list_all(self) -> list[str]:
        return sorted(self._entries)

    def build(self, name: str, **kwargs) -> nn.Module:
        return self.get(name)(**kwargs)


LIBRARY = LossLibrary()
LIBRARY.register("masked_l1", MaskedL1Loss)
LIBRARY.register("masked_l1_gradient", MaskedL1GradientLoss)
LIBRARY.register("masked_huber", MaskedHuberLoss)


__all__ = [
    "MaskedL1Loss",
    "MaskedL1GradientLoss",
    "MaskedHuberLoss",
    "LossLibrary",
    "LIBRARY",
    "building_mask",
]



