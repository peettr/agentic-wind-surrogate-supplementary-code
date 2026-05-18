"""Process-wide ModelRegistry for auto_v3 surrogates.

Built-in architectures are registered at import time. Codegen-produced models are
added dynamically via :meth:`ModelRegistry.register_from_path` after preflight.
"""
from __future__ import annotations

import importlib
from typing import Callable

from .base import BaseSurrogate
from .cno import CNO
from .convnext_v2_unet import ConvNeXtV2UNet
from .ffno import FFNO
from .fno_v3 import FNO2d
from .ufno import UFNO
from .afno_block import AFNOBlock, AFNOBottleneck
from .swin_unetr import SwinUNETR
from .unet_afno import UNetAFNO
from .unet_v3 import UNet
from .unet_sdf_7level import UNetSDF
from .unet_v2_baseline import UNetV2Baseline
from .attention_gate_unet import AttentionGateUNet
from .nafnet import NAFNet
from .sac_unet import SACUNet
from .dilated_unet import DilatedUNet
from .cbam_unet import CBAMUNet
from .dcn_unet import DCNUNet
from .kan_unet import KANUNet
from .hrnet import HRNetSurrogate
from .transolver import Transolver
from .umamba import UMamba
from .mamba2d import Mamba2D
from .quadmamba import QuadMamba
from .hrformer import HRFormer
from .perceiver import PerceiverIO
from .cnn_deeponet import CNNDeepONet


class ModelRegistry:
    """name → constructor mapping for surrogate architectures."""

    def __init__(self) -> None:
        self._entries: dict[str, Callable[..., BaseSurrogate]] = {}

    def register(self, name: str, ctor: Callable[..., BaseSurrogate]) -> None:
        if name in self._entries:
            raise KeyError(f"Architecture '{name}' already registered")
        self._entries[name] = ctor

    def register_from_path(self, name: str, module_path: str, class_name: str) -> None:
        """Import ``module_path`` and register its ``class_name`` under ``name``."""
        mod = importlib.import_module(module_path)
        ctor = getattr(mod, class_name)
        self.register(name, ctor)

    def get(self, name: str) -> Callable[..., BaseSurrogate]:
        if name not in self._entries:
            raise KeyError(
                f"Unknown architecture '{name}'. Available: {self.list_all()}"
            )
        return self._entries[name]

    def list_all(self) -> list[str]:
        return sorted(self._entries)

    def build(self, name: str, **kwargs) -> BaseSurrogate:
        return self.get(name)(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


# Default process-wide registry (populated with built-ins).
REGISTRY = ModelRegistry()
REGISTRY.register("unet_v2_baseline", UNetV2Baseline)
REGISTRY.register("unet_v3", lambda **kw: UNet(**{"depth": 7, "n_c": 16, **kw}))
REGISTRY.register("unet_v3_5level", lambda **kw: UNet(**{"depth": 5, "n_c": 16, **kw}))
REGISTRY.register("unet_v3_6level", lambda **kw: UNet(**{"depth": 6, "n_c": 16, **kw}))
REGISTRY.register("unet_v3_7level", lambda **kw: UNet(**{"depth": 7, "n_c": 16, **kw}))
REGISTRY.register("cno", CNO)
REGISTRY.register("convnext_v2_unet", ConvNeXtV2UNet)
REGISTRY.register("swin_unetr", SwinUNETR)
REGISTRY.register("unet_afno", UNetAFNO)
REGISTRY.register("ffno", FFNO)
REGISTRY.register("fno_v3", FNO2d)
REGISTRY.register("ufno", UFNO)
REGISTRY.register("unet_sdf_7level", lambda **kw: UNetSDF(**{"in_channels": 3, "base_ch": 64, **kw}))

# Tier A: UNet variants
REGISTRY.register("attention_gate_unet", AttentionGateUNet)
REGISTRY.register("nafnet", NAFNet)
REGISTRY.register("sac_unet", SACUNet)
REGISTRY.register("dilated_unet", DilatedUNet)
REGISTRY.register("cbam_unet", CBAMUNet)
REGISTRY.register("dcn_unet", DCNUNet)
REGISTRY.register("kan_unet", KANUNet)
REGISTRY.register("hrnet", HRNetSurrogate)

# Tier B: New architectures
REGISTRY.register("transolver", Transolver)
REGISTRY.register("umamba", UMamba)
REGISTRY.register("mamba2d", Mamba2D)
REGISTRY.register("quadmamba", QuadMamba)
REGISTRY.register("hrformer", HRFormer)
REGISTRY.register("perceiver_io", PerceiverIO)
REGISTRY.register("perceiver", PerceiverIO)  # alias used by planner/scouts
REGISTRY.register("cnn_deeponet", CNNDeepONet)

# Tier C: Operator methods
from .fno2d import FNO2d as FNO2dNew
from .afno import AFNO
from .cno import CNO as CNOv2
from .uno import UNO
REGISTRY.register("fno2d", FNO2dNew)
REGISTRY.register("afno", AFNO)
REGISTRY.register("cno_v2", CNOv2)
REGISTRY.register("uno", UNO)

# Tier D: Hybrid architectures
from .dilated_fno import DilatedFNO
from .sac_mamba import SACMamba
from .mamba_attention import MambaAttention
from .hrdcn import HRDCN
from .fourier_unet import FourierUNet
from .transolver_lite import TransolverLite
from .dilated_hrformer import DilatedHRFormer
from .fno_encoder_decoder import FNOEncoderDecoder
from .attention_mamba import AttentionMamba
from .multiscale_conv import MultiScaleConv
from .residual_spectral import ResidualSpectralNet
REGISTRY.register("dilated_fno", DilatedFNO)
REGISTRY.register("sac_mamba", SACMamba)
REGISTRY.register("mamba_attention", MambaAttention)
REGISTRY.register("hrdcn", HRDCN)
REGISTRY.register("fourier_unet", FourierUNet)
REGISTRY.register("transolver_lite", TransolverLite)
REGISTRY.register("dilated_hrformer", DilatedHRFormer)
REGISTRY.register("fno_encoder_decoder", FNOEncoderDecoder)
REGISTRY.register("attention_mamba", AttentionMamba)
REGISTRY.register("multiscale_conv", MultiScaleConv)
REGISTRY.register("residual_spectral", ResidualSpectralNet)


__all__ = ["BaseSurrogate", "UNet", "FNO2d", "ModelRegistry", "REGISTRY"]
