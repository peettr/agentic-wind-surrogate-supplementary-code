"""AI-driven experiment suggester for Hybrid.

Uses V3 baseline results to suggest new experiments via:
  1. Factor analysis: Identify which hyperparameters matter most
  2. Architecture ranking: Focus on best-performing families
  3. Gap filling: Explore under-sampled regions of the search space
  4. Transfer: Apply SDF input to top height-only models

The suggester can run in two modes:
  - rule_based: Deterministic suggestions based on V3 findings (no AI call)
  - ai_driven: Uses Claude/GPT to generate novel suggestions (future)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from explorer import ExperimentConfig, ExperimentResult, summarize_results


# V3 validated findings (hardcoded as rules)
V3_BEST_CONFIGS = {
    "sdf_3ch": {
        "arch_name": "unet_v2_baseline",
        "n_c": 32,
        "depth": 7,
        "loss_name": "masked_l1",
        "lr": 1e-3,
        "input_features": "height_sdf_normal",
        "use_ema": False,
        "epochs": 200,
    },
    "orthogonal exploratory sweep_champion": {
        "arch_name": "unet_v2_baseline",
        "n_c": 32,
        "loss_name": "masked_l1",
        "lr": 5e-4,
        "use_ema": True,
        "ema_decay": 0.999,
        "augmentation": False,
        "epochs": 200,
    },
    "dilated_unet": {
        "arch_name": "dilated_unet",
        "n_c": 32,
        "depth": 7,
        "loss_name": "masked_l1",
        "lr": 5e-4,
        "epochs": 200,
    },
}

# Architectures to avoid (V3 confirmed poor)
V3_CLOSED_DIRECTIONS = {
    "fno2d": "R=0.33, spectral mixing unsuitable",
    "afno": "R=0.056, even after bug fix",
    "perceiver": "R=-0.05, resolution collapse",
    "cnn_deeponet": "R=-3.28, constant output",
    "nafnet": "R~0.01, too few params (5M)",
    "transolver_lite": "R=-3.28, dead code (no positional embedding)",
}

# Architectures with potential (V3 mid-range, worth exploring with SDF)
V3_PROMISING_ARCHS = [
    "dilated_unet",      # R=0.704 height-only, +SDF expected ~0.74
    "unet_v2_baseline",  # R=0.680 height-only, +SDF = 0.724
    "quadmamba",         # R=0.680, best Mamba variant
    "mamba2d",           # R=0.684, needs CUDA scan
    "umamba",            # R=0.683, now with real SSM
    "ag_unet",           # R=0.664, attention gates + SDF
    "sac_unet",          # R=0.661, now with real SAC
    "hrdcn",             # R=0.345, now with real DCN
    "fourier_unet",      # R=0.398, now with correct FFT pad
    "cno",               # R=0.650, worth testing with SDF
]


@dataclass
class SuggestionBatch:
    """A batch of suggested experiments for the next round."""
    round_num: int
    strategy: str  # "sdf_transfer", "architecture_explore", "hyperparam_finetune"
    experiments: list[ExperimentConfig]
    rationale: str

    def save(self, path: Path):
        data = {
            "round_num": self.round_num,
            "strategy": self.strategy,
            "rationale": self.rationale,
            "experiments": [vars(e) for e in self.experiments],
        }
        path.write_text(json.dumps(data, indent=2))


def suggest_sdf_transfer(round_num: int = 1) -> SuggestionBatch:
    """Apply SDF 3ch input to top height-only models from V3.
    
    This is the single highest-value experiment batch:
    SDF input gave +0.044 R on UNet, so applying it to DilatedUNet,
    QuadMamba, Mamba2D, UMamba, AG-UNet should yield similar gains.
    """
    archs_to_test = [
        ("dilated_unet", 32, 7, "masked_l1", 1e-3),
        ("dilated_unet", 32, 7, "masked_l1_gradient", 1e-3),
        ("dilated_unet", 32, 7, "masked_l1", 5e-4),
        ("quadmamba", 32, 7, "masked_l1_gradient", 1e-3),
        ("mamba2d", 32, 7, "masked_l1_gradient", 1e-3),
        ("umamba", 32, 7, "masked_l1_gradient", 1e-3),
        ("ag_unet", 32, 7, "masked_l1_gradient", 1e-3),
        ("sac_unet", 32, 7, "masked_l1", 1e-3),
        ("cno", 32, 7, "masked_l1_gradient", 5e-4),
        ("unet_v2_baseline", 32, 7, "masked_l1", 5e-4),  # with EMA
    ]

    experiments = []
    for arch, nc, depth, loss, lr in archs_to_test:
        experiments.append(ExperimentConfig(
            arch_name=arch,
            n_c=nc,
            depth=depth,
            loss_name=loss,
            lr=lr,
            input_features="height_sdf_normal",
            use_ema=(lr == 5e-4),  # EMA with low lr
            ema_decay=0.999,
            epochs=200,
        ))

    return SuggestionBatch(
        round_num=round_num,
        strategy="sdf_transfer",
        experiments=experiments,
        rationale=(
            "V3 showed SDF 3ch input gives +0.044 R on UNet (0.680 to 0.724). "
            "Apply the same input to the top 9 height-only models. "
            "Expected best: DilatedUNet+SDF R~0.735-0.745."
        ),
    )


def suggest_architecture_explore(round_num: int, prior_results: list[ExperimentResult]) -> SuggestionBatch:
    """Suggest new architectures or variants based on V3 gaps.
    
    Focus on:
    1. Fixed V3 models (UMamba, SAC-UNet, HRDCN, FourierUNet) with SDF
    2. Multi-scale variants of best architectures
    3. Hybrid combinations (DilatedUNet + attention, etc.)
    """
    experiments = []

    # Fixed models that haven't been tested with SDF yet
    fixed_models = ["umamba", "sac_unet", "hrdcn", "fourier_unet"]
    for arch in fixed_models:
        experiments.append(ExperimentConfig(
            arch_name=arch,
            n_c=32,
            depth=7,
            loss_name="masked_l1",
            lr=1e-3,
            input_features="height_sdf_normal",
            epochs=200,
        ))

    # DilatedUNet variants with different dilation rates
    for dilation in [1, 3, 4]:
        experiments.append(ExperimentConfig(
            arch_name="dilated_unet",
            n_c=32,
            depth=7,
            loss_name="masked_l1",
            lr=1e-3,
            input_features="height_sdf_normal",
            epochs=200,
        ))

    # Longer training for top models
    experiments.append(ExperimentConfig(
        arch_name="dilated_unet",
        n_c=32,
        depth=7,
        loss_name="masked_l1",
        lr=5e-4,
        use_ema=True,
        ema_decay=0.999,
        input_features="height_sdf_normal",
        epochs=1000,
    ))

    return SuggestionBatch(
        round_num=round_num,
        strategy="architecture_explore",
        experiments=experiments,
        rationale=(
            "Explore fixed V3 models with SDF input, test DilatedUNet "
            "dilation variants, and run a 1000-epoch long training."
        ),
    )


def suggest_hyperparam_finetune(round_num: int, prior_results: list[ExperimentResult]) -> SuggestionBatch:
    """Fine-tune hyperparameters around the best configurations.
    
    Based on V3 orthogonal exploratory sweep findings:
    - l1_gradient loss is competitive with l1
    - lr=5e-4 and 1e-3 are both viable
    - EMA helps at lr=5e-4
    - cosine scheduler has mixed results
    """
    experiments = []
    best_arch = "dilated_unet"  # Will be dynamically determined from prior_results

    # Find the best configuration from prior results
    completed = [r for r in prior_results if r.status == "completed" and r.score > 0]
    if completed:
        best = max(completed, key=lambda r: r.score)
        best_arch = best.config.arch_name

    # Fine-tune around best
    for loss in ["masked_l1", "masked_l1_gradient"]:
        for lr in [3e-4, 5e-4, 7e-4, 1e-3]:
            for ema in [True, False]:
                experiments.append(ExperimentConfig(
                    arch_name=best_arch,
                    n_c=32,
                    depth=7,
                    loss_name=loss,
                    lr=lr,
                    use_ema=ema,
                    ema_decay=0.999 if ema else 0.999,
                    input_features="height_sdf_normal",
                    scheduler="cosine" if lr < 1e-3 else None,
                    epochs=200,
                ))

    return SuggestionBatch(
        round_num=round_num,
        strategy="hyperparam_finetune",
        experiments=experiments[:16],  # Cap at 16 per round
        rationale=f"Fine-tune loss/lr/EMA around best arch={best_arch} with SDF input.",
    )


def generate_suggestions(
    mode: str = "sdf_transfer",
    round_num: int = 1,
    prior_results: Optional[list[ExperimentResult]] = None,
) -> SuggestionBatch:
    """Main entry point for suggestion generation."""
    if prior_results is None:
        prior_results = []

    if mode == "sdf_transfer":
        return suggest_sdf_transfer(round_num)
    elif mode == "architecture_explore":
        return suggest_architecture_explore(round_num, prior_results)
    elif mode == "hyperparam_finetune":
        return suggest_hyperparam_finetune(round_num, prior_results)
    else:
        raise ValueError(f"Unknown suggestion mode: {mode}")


if __name__ == "__main__":
    from explorer import load_v3_baseline

    print("Hybrid Suggester - Generating SDF transfer suggestions...")
    batch = generate_suggestions("sdf_transfer", round_num=1)
    print(f"Strategy: {batch.strategy}")
    print(f"Rationale: {batch.rationale}")
    print(f"\n{len(batch.experiments)} experiments suggested:")
    for i, exp in enumerate(batch.experiments):
        print(f"  {i+1}. {exp.arch_name} | loss={exp.loss_name} lr={exp.lr} input={exp.input_features}")

    # Save
    out_dir = Path(__file__).parent.parent / "configs"
    out_dir.mkdir(exist_ok=True)
    batch.save(out_dir / "suggestion_round1.json")
    print(f"\nSaved to configs/suggestion_round1.json")




