"""Hybrid Explorer - AI-driven experiment suggestion and evaluation loop.

The Explorer is the core of Sequential: instead of fixed orthogonal grids (V3's L18),
it uses prior results to suggest new experiments, evaluates them, and feeds
the results back into the suggestion loop.

V3 results are NOT loaded. Sequential discovers everything from scratch.
"""
from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Project root = parent of explorer/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SHARED_DIR = PROJECT_ROOT / "shared"


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    arch_name: str
    n_c: int = 16
    depth: int = 7
    loss_name: str = "masked_l1"
    lr: float = 1e-3
    batch_size: int = 16
    scheduler: Optional[str] = None
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = None
    use_ema: bool = False
    ema_decay: float = 0.999
    augmentation: bool = False
    input_features: str = "height"
    epochs: int = 200
    seed: int = 1

    def config_hash(self) -> str:
        s = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:8]

    def to_train_config(self, results_dir: Path) -> dict:
        return {
            "arch_name": self.arch_name,
            "arch_kwargs": {"depth": self.depth, "n_c": self.n_c},
            "data_dir": "<PROJECT_HPC_ROOT>/data",
            "results_dir": str(results_dir),
            "split_manifest_path": "<PROJECT_HPC_ROOT>/data/split_manifest.json",
            "loss_name": self.loss_name,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "scheduler": self.scheduler,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "augmentation": self.augmentation,
            "input_features": self.input_features,
            "epochs": self.epochs,
            "seed": self.seed,
        }


@dataclass
class ExperimentResult:
    """Result from a completed experiment."""
    config: ExperimentConfig
    r2_median: float = float("nan")
    r2_global: float = float("nan")
    mae_median: float = float("nan")
    nmae_median: float = float("nan")
    peak_vram_gb: float = 0.0
    wall_time_sec: float = 0.0
    gpu: str = ""
    cluster_id: int = 0
    status: str = "pending"

    @property
    def score(self) -> float:
        if not (self.r2_median != self.r2_median):
            return self.r2_median
        if not (self.r2_global != self.r2_global):
            return self.r2_global
        return -1.0


def load_v3_baseline() -> list[ExperimentResult]:
    """Load ONLY the original V3 baseline result (frozen, Sequential cannot modify).

    What is disclosed:
    - unet_v2_baseline model code (frozen, Sequential cannot modify)
    - orthogonal exploratory sweep baseline run (run_00): R2=0.680 at 200ep, lr=5e-4, n_c=16, height-only

    What is NOT disclosed:
    - orthogonal exploratory sweep other 18 runs (the tuning results)
    - Any other V3 experiment results
    """
    baseline_result = ExperimentResult(
        config=ExperimentConfig(
            arch_name="unet_v2_baseline",
            n_c=16, depth=7, loss_name="masked_l1", lr=5e-4,
            batch_size=16, scheduler=None, weight_decay=0,
            gradient_clip=None, use_ema=False, augmentation=False,
            input_features="height", epochs=200, seed=1,
        ),
        r2_median=0.680,  # orthogonal exploratory sweep run_00: 200ep, lr=5e-4
        mae_median=0.092,
        peak_vram_gb=0, wall_time_sec=0, gpu="",
        status="completed",
    )
    return [baseline_result]


def summarize_results(results: list[ExperimentResult]) -> dict:
    """Generate a summary of all results for AI suggestion."""
    by_arch: dict[str, list[ExperimentResult]] = {}
    for r in results:
        by_arch.setdefault(r.config.arch_name, []).append(r)

    summary = {
        "total_experiments": len(results),
        "completed": sum(1 for r in results if r.status == "completed"),
        "architectures": {},
        "top_10": [],
    }

    for arch, arch_results in sorted(by_arch.items()):
        scores = [r.score for r in arch_results if r.status == "completed"]
        if scores:
            summary["architectures"][arch] = {
                "count": len(scores),
                "best_r2": max(scores),
                "mean_r2": sum(scores) / len(scores),
            }

    completed = [r for r in results if r.status == "completed"]
    completed.sort(key=lambda r: r.score, reverse=True)
    for r in completed[:10]:
        summary["top_10"].append({
            "arch": r.config.arch_name,
            "r2": round(r.score, 4),
            "loss": r.config.loss_name,
            "lr": r.config.lr,
            "n_c": r.config.n_c,
            "ema": r.config.use_ema,
            "input": r.config.input_features,
        })

    return summary




