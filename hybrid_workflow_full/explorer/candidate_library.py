"""Candidate Library for Hybrid Explorer.

The library defines the SEARCH SPACE for the Explorer:
  1. Model catalog: ALL architectures from V3 (code + architecture details only)
  2. Hyperparameter space: ranges and discrete values for each factor

CRITICAL CONSTRAINT: No V3 results (R2, rankings, priorities) are included.
The Explorer has NO prior knowledge about which models or HPs are better.
Sequential discovers everything through its own experiments.

What transfers from V3:
  - Model implementation code (shared/models/*.py)
  - HP ranges that were found to be valid (not divergent)
  - Architecture details (param count, depth options, etc.)

What does NOT transfer:
  - V3 R2 values
  - Champion/competitive/weak/failed status labels
  - Any ranking or priority information
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ======================================================================
# Model Catalog - ALL 42 models from V3, no performance data
# ======================================================================

@dataclass
class ModelSpec:
    """Specification of a candidate model. No V3 results included."""
    name: str
    category: str  # baseline, tier_a, tier_b, tier_c, tier_d
    params_million: float  # parameter count (informational only)
    depth_options: list[int] = field(default_factory=lambda: [7])
    n_c_options: list[int] = field(default_factory=lambda: [16, 32])
    input_channels: int = 1  # default height-only; Explorer can try SDF (3ch)
    enabled: bool = True
    notes: str = ""  # architecture description, not performance


MODEL_CATALOG = [
    # === Baselines ===
    ModelSpec("unet_v2_baseline", "baseline", 34.6,
              depth_options=[5, 6, 7], n_c_options=[8, 16, 32, 48],
              notes="7-level UNet, V3 reference. 6 ConvBlock encoder + decoder + skip concat."),
    ModelSpec("unet_v3", "baseline", 34.6,
              depth_options=[5, 6, 7], n_c_options=[16, 32],
              notes="V3 variant of baseline UNet."),
    ModelSpec("unet_sdf_7level", "baseline", 34.6,
              depth_options=[7], n_c_options=[16, 32], input_channels=3,
              notes="UNet with 3-channel input (height, SDF, normal angle)."),
    ModelSpec("unet_afno", "baseline", 11.9,
              depth_options=[5, 6, 7], n_c_options=[16, 24, 32],
              notes="UNet with AFNO spectral bottleneck at deepest level."),

    # === Tier A: UNet variants ===
    ModelSpec("attention_gate_unet", "tier_a", 503.0,
              depth_options=[7], n_c_options=[32],
              notes="Schlemper attention gates on every skip connection."),
    ModelSpec("cbam_unet", "tier_a", 501.0,
              depth_options=[7], n_c_options=[32],
              notes="CBAM (channel + spatial) attention on skips."),
    ModelSpec("dilated_unet", "tier_a", 498.0,
              depth_options=[7], n_c_options=[32, 48],
              notes="DilatedConvBlock with dilation=2. Effective 5x5 receptive field at same params."),
    ModelSpec("dcn_unet", "tier_a", 135.0,
              depth_options=[7], n_c_options=[16, 32],
              notes="Deformable convolution (approximated via dense warp via grid_sample)."),
    ModelSpec("sac_unet", "tier_a", 134.0,
              depth_options=[7], n_c_options=[16, 32],
              notes="Spatial adaptive conv: standard conv + per-pixel scale/shift modulation."),
    ModelSpec("nafnet", "tier_a", 5.0,
              depth_options=[7], n_c_options=[8, 16],
              notes="SimpleGate blocks, no activation. Very small (5M params)."),
    ModelSpec("kan_unet", "tier_a", 22.0,
              depth_options=[7], n_c_options=[16, 32],
              notes="Kolmogorov-Arnold Network blocks in UNet shell."),
    ModelSpec("hrnet", "tier_a", 20.0,
              depth_options=[4], n_c_options=[16, 32],
              notes="Multi-resolution parallel streams. Stride-4 stem limits internal res to 160x160."),

    # === Tier B: New architectures ===
    ModelSpec("transolver", "tier_b", 121.0,
              depth_options=[6], n_c_options=[16, 32],
              notes="Transformer with solver-style attention. V3 runs all failed (timeout/killed)."),
    ModelSpec("umamba", "tier_b", 288.0,
              depth_options=[7], n_c_options=[32],
              notes="UNet + bidirectional selective SSM in bottleneck. 4-direction EMA scan."),
    ModelSpec("mamba2d", "tier_b", 423.0,
              depth_options=[7], n_c_options=[32],
              notes="4-direction serial EMA scan (Python loop). Slow wall time."),
    ModelSpec("quadmamba", "tier_b", 338.0,
              depth_options=[7], n_c_options=[32],
              notes="Four-quadrant EMA scan with shared projection weights."),
    ModelSpec("hrformer", "tier_b", 13.0,
              depth_options=[4], n_c_options=[16, 32],
              notes="Window self-attention HRNet. Stride-2 stem."),
    ModelSpec("swin_unetr", "tier_b", 12.7,
              depth_options=[4], n_c_options=[16, 32],
              notes="Swin Transformer encoder + conv decoder. Stride-4 patch embed."),
    ModelSpec("perceiver", "tier_b", 7.0,
              depth_options=[6], n_c_options=[16, 32],
              notes="Latent cross-attention. 8-fold stride conv + 32x32 latent array."),
    ModelSpec("cnn_deeponet", "tier_b", 2.0,
              depth_options=[6], n_c_options=[16, 32],
              notes="CNN branch (global avg pool) + MLP trunk. Very small."),

    # === Tier C: Neural operators ===
    ModelSpec("fno2d", "tier_c", 12.1,
              depth_options=[4], n_c_options=[16, 24, 32],
              notes="2D Fourier Neural Operator. Truncated-mode spectral mixing."),
    ModelSpec("afno", "tier_c", 11.9,
              depth_options=[4], n_c_options=[16, 24, 32],
              notes="Adaptive FNO. Block-diagonal complex MLP in FFT domain."),
    ModelSpec("cno", "tier_c", 14.2,
              depth_options=[4, 5], n_c_options=[16, 32],
              notes="Convolutional Neural Operator. CNOBlocks with channel lifting."),
    ModelSpec("uno", "tier_c", 20.0,
              depth_options=[4], n_c_options=[16, 32],
              notes="U-shaped Neural Operator."),
    ModelSpec("transolver_lite", "tier_c", 2.5,
              depth_options=[4, 6], n_c_options=[16, 32],
              notes="Simplified Transolver. Very small."),

    # === Tier D: Hybrids ===
    ModelSpec("dilated_fno", "tier_d", 15.0,
              depth_options=[4], n_c_options=[16, 24],
              notes="Multi-rate dilated conv (1,2,4 parallel) + spectral skip."),
    ModelSpec("sac_mamba", "tier_d", 16.6,
              depth_options=[4], n_c_options=[16, 24],
              notes="SAC + Mamba bottleneck hybrid."),
    ModelSpec("hrdcn", "tier_d", 15.0,
              depth_options=[4], n_c_options=[16, 24],
              notes="HRNet + DCNv2 (deformable conv)."),
    ModelSpec("fourier_unet", "tier_d", 8.6,
              depth_options=[4], n_c_options=[16, 24],
              notes="UNet with FFT-domain upsampling in decoder."),
    ModelSpec("dilated_hrformer", "tier_d", 18.9,
              depth_options=[4], n_c_options=[16, 24],
              notes="Dilated conv + HRFormer window attention."),
    ModelSpec("fno_encoder_decoder", "tier_d", 14.9,
              depth_options=[4], n_c_options=[16, 24],
              notes="FNO block in encoder-decoder shell."),
    ModelSpec("mamba_attention", "tier_d", 16.0,
              depth_options=[4], n_c_options=[16, 24],
              notes="Attention + Mamba hybrid variant."),
    ModelSpec("multiscale_conv", "tier_d", 8.4,
              depth_options=[4], n_c_options=[16, 24],
              notes="4-rate dilated conv (1,2,4,8) + SE channel attention."),
    ModelSpec("residual_spectral", "tier_d", 6.4,
              depth_options=[4], n_c_options=[16, 24],
              notes="Residual spectral convolution."),
]


# ======================================================================
# Hyperparameter Search Space (ranges, not presets)
# ======================================================================

@dataclass
class HPSpace:
    """Hyperparameter search space. Explorer samples from these ranges.
    
    These are RECOMMENDATIONS, not hard constraints.
    The Explorer may suggest values outside this space.
    """
    # Architecture â€” depth removed, each model uses its own default
    # n_c: only 16 and 32 (V3 validated range)
    n_c: list[int] = field(default_factory=lambda: [16, 32])

    # Loss
    loss_name: list[str] = field(default_factory=lambda: [
        "masked_l1", "masked_l1_gradient", "masked_huber"
    ])

    # Optimizer â€” lr baseline is 1e-3 (V3 DEFAULT_LR), Explorer may suggest others
    lr: list[float] = field(default_factory=lambda: [1e-3])
    weight_decay: list[float] = field(default_factory=lambda: [0.0, 1e-5, 1e-4])

    # Scheduler
    scheduler: list[Optional[str]] = field(default_factory=lambda: [None, "cosine"])

    # Regularization
    gradient_clip: list[Optional[float]] = field(default_factory=lambda: [None, 0.5])
    use_ema: list[bool] = field(default_factory=lambda: [False, True])
    ema_decay: list[float] = field(default_factory=lambda: [0.999])

    # Augmentation (V3 showed this breaks physics, Explorer should discover this)
    augmentation: list[bool] = field(default_factory=lambda: [False, True])

    # Input representation
    input_features: list[str] = field(default_factory=lambda: [
        "height",            # 1 channel
        "height_sdf",        # 2 channels
        "height_sdf_normal"  # 3 channels
    ])

# Training: ordinary candidates use batch_size=16 fixed. Lower batch sizes are
# resource_probe/OOM-repair feasibility paths only, not leaderboard candidates.
    epochs: list[int] = field(default_factory=lambda: [20, 200, 1000])

    def sample_random(self, rng: Optional[random.Random] = None) -> dict:
        """Sample a random configuration from the space."""
        import random as _rng
        r = rng or _rng
        return {
            "n_c": r.choice(self.n_c),
            "loss_name": r.choice(self.loss_name),
            "lr": r.choice(self.lr),
            "weight_decay": r.choice(self.weight_decay),
            "scheduler": r.choice(self.scheduler),
            "gradient_clip": r.choice(self.gradient_clip),
            "use_ema": r.choice(self.use_ema),
            "ema_decay": r.choice(self.ema_decay),
            "augmentation": r.choice(self.augmentation),
            "input_features": r.choice(self.input_features),
            "batch_size": 16,  # fixed (A1)
            "epochs": r.choice(self.epochs),
        }

    def get_baseline_config(self) -> dict:
        """V3 baseline configuration (starting point for exploration)."""
        return {
            "n_c": 16,
            "loss_name": "masked_l1",
            "lr": 1e-3,  # V3 DEFAULT_LR baseline
            "weight_decay": 0.0,
            "scheduler": None,
            "gradient_clip": None,
            "use_ema": False,
            "ema_decay": 0.999,
            "augmentation": False,
            "input_features": "height",
            "batch_size": 16,
            "epochs": 200,
        }


# ======================================================================
# Library facade
# ======================================================================

class CandidateLibrary:
    """The Explorer's reference library for experiment suggestions.

    Contains model details and HP search space only.
    NO V3 performance data - Sequential discovers everything fresh.
    """

    def __init__(self):
        self.models = {m.name: m for m in MODEL_CATALOG}
        self.hp_space = HPSpace()

    def get_enabled_models(self, category: Optional[str] = None) -> list[ModelSpec]:
        models = [m for m in self.models.values() if m.enabled]
        if category:
            models = [m for m in models if m.category == category]
        return models

    def get_all_models(self) -> list[ModelSpec]:
        return list(self.models.values())

    def disable_model(self, name: str, reason: str = ""):
        if name in self.models:
            self.models[name].enabled = False
            if reason:
                self.models[name].notes += f" [DISABLED: {reason}]"

    def to_dict(self) -> dict:
        return {
            "models": [
                {"name": m.name, "category": m.category,
                 "params_million": m.params_million,
                 "depth_options": m.depth_options,
                 "n_c_options": m.n_c_options,
                 "input_channels": m.input_channels,
                 "enabled": m.enabled, "notes": m.notes}
                for m in MODEL_CATALOG
            ],
            "hp_space": {
                "loss_name": self.hp_space.loss_name,
                "lr": self.hp_space.lr,
                "weight_decay": self.hp_space.weight_decay,
                "scheduler": [s for s in self.hp_space.scheduler],
                "gradient_clip": self.hp_space.gradient_clip,
                "use_ema": self.hp_space.use_ema,
                "augmentation": self.hp_space.augmentation,
                "input_features": self.hp_space.input_features,
                "epochs": self.hp_space.epochs,
                "n_c": self.hp_space.n_c,
                "batch_size": 16,  # fixed
            },
        }

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    lib = CandidateLibrary()
    print(f"Candidate Library: {len(lib.models)} models, HP space with {len(lib.hp_space.lr)} lr values\n")

    print("=== All Models (no performance data) ===")
    for m in MODEL_CATALOG:
        print(f"  {m.name:25s} | {m.category:10s} | {m.params_million:6.1f}M | in_ch={m.input_channels} | enabled={m.enabled}")

    # Save library
    out = Path(__file__).parent.parent / "configs" / "candidate_library.json"
    lib.save(out)
    print(f"\nLibrary saved to {out}")




