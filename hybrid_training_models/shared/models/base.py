"""Abstract base class for every surrogate architecture.

The I/O contract is part of the Appendix-A locked parameters. Every architecture
registered with ModelRegistry must subclass :class:`BaseSurrogate` and accept the
fixed input shape `(B, 1, 640, 640)` while producing an identically-shaped,
non-negative output (final activation is ReLU).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


EXPECTED_INPUT_SHAPE = (1, 640, 640)   # (C, H, W); batch dim flexible
EXPECTED_OUTPUT_SHAPE = (1, 640, 640)


class BaseSurrogate(nn.Module, ABC):
    """Abstract surrogate model with fixed I/O contract."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape ``(B, 1, 640, 640)``.
        Returns:
            prediction tensor of shape ``(B, 1, 640, 640)`` with non-negative values.
        """

    def check_shapes(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Validate the I/O contract at runtime (invoked by preflight)."""
        if x.shape[1:] != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"Input shape mismatch: expected (B, 1, 640, 640), got {tuple(x.shape)}"
            )
        if y.shape[1:] != EXPECTED_OUTPUT_SHAPE:
            raise ValueError(
                f"Output shape mismatch: expected (B, 1, 640, 640), got {tuple(y.shape)}"
            )
