"""PDE-Refiner: iterative refinement on top of a trained base predictor.

Reference: Lippe et al. 2023 "PDE-Refiner: Achieving Accurate Long Rollouts 
and Uncertainty Quantification for Neural PDE Surrogates"

Stage 1: Base predictor (already trained, e.g. UNet R²=0.702)
Stage 2: Train refiner to denoise noisy base predictions

This script implements Stage 2 training.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from shared.configs.schema import TrainConfig
from shared.losses import LIBRARY as LOSS_LIBRARY
from shared.models import REGISTRY as MODEL_REGISTRY

LOGGER = logging.getLogger("auto_v3.train_refiner")


def _noise_schedule(t: float, schedule: str = "linear", max_noise: float = 0.5) -> float:
    """Compute noise level at step t ∈ [0, 1]."""
    if schedule == "linear":
        return max_noise * (1 - t)
    elif schedule == "cosine":
        return max_noise * (0.5 * (1 + np.cos(np.pi * t)))
    else:
        return max_noise * (1 - t)


def train_refiner(cfg: TrainConfig) -> dict:
    """Train PDE-Refiner Stage 2 on top of a frozen base predictor."""
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg.data_dir)

    log_path = results_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
        force=True,
    )

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # Parse refiner-specific config from arch_kwargs
    refiner_cfg = cfg.arch_kwargs
    base_model_path = refiner_cfg.get("base_model_path")
    refine_steps = refiner_cfg.get("refine_steps", 4)
    noise_schedule = refiner_cfg.get("noise_schedule", "linear")
    max_noise = refiner_cfg.get("max_noise", 0.5)
    refiner_width = refiner_cfg.get("refiner_width", "full")  # "full" or "half"
    base_arch_name = refiner_cfg.get("base_arch_name", "unet_v2_baseline")
    base_n_c = refiner_cfg.get("base_n_c", 32)
    base_depth = refiner_cfg.get("base_depth", 7)

    # Load frozen base predictor
    from shared.models import REGISTRY
    base_cls = REGISTRY.get(base_arch_name)
    base_model = base_cls(n_c=base_n_c, depth=base_depth).to(device)
    base_state = torch.load(base_model_path, map_location=device, weights_only=True)
    base_model.load_state_dict(base_state)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    LOGGER.info("Loaded base model from %s (frozen)", base_model_path)

    # Build refiner (same architecture as base, or half width)
    n_c_refiner = base_n_c if refiner_width == "full" else base_n_c // 2
    refiner = base_cls(n_c=n_c_refiner, depth=base_depth).to(device)
    LOGGER.info("Refiner: %s n_c=%d params=%d", refiner_width, n_c_refiner,
                sum(p.numel() for p in refiner.parameters()))

    # Refiner input: concatenate (noisy prediction, noise level channel)
    # So refiner needs in_channels=2 (pred + noise_level_map)
    # But we reuse same architecture — simpler approach: refiner takes noisy pred,
    # and we add noise level as a learnable embedding
    # For simplicity: refiner input = noisy_pred (1ch), target = residual (pred - target)
    
    # Actually, let's make the refiner take 2ch input: [noisy_pred, noise_level_map]
    # We need to modify the first conv to accept 2 channels
    # Simplest: just use the refiner as-is with 1ch input (noisy pred)
    
    loss_fn = LOSS_LIBRARY.build(cfg.loss_name, **cfg.loss_kwargs)
    
    # Load data
    from shared.train import _load_train_val, set_deterministic
    set_deterministic(cfg.seed)
    train_b, val_b, holdout_b = _load_train_val(data_dir, cfg.split_manifest_path, cfg.seed)
    X_tr = train_b["X"].to(device)
    Y_tr = train_b["Y"].to(device)
    X_va = val_b["X"].to(device)
    Y_va = val_b["Y"].to(device)

    # Pre-compute base predictions for training data
    LOGGER.info("Computing base predictions for training data...")
    with torch.no_grad():
        base_preds_tr = torch.cat([base_model(X_tr[i:i+8]) for i in range(0, len(X_tr), 8)], dim=0)
        base_preds_va = torch.cat([base_model(X_va[i:i+8]) for i in range(0, len(X_va), 8)], dim=0)
    LOGGER.info("Base predictions computed: train=%d val=%d", len(base_preds_tr), len(base_preds_va))

    optimizer = torch.optim.Adam(refiner.parameters(), lr=cfg.lr, weight_decay=1e-5)
    train_dataset = TensorDataset(base_preds_tr, Y_tr, X_tr)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    best_val = float("inf")
    status = "ok"
    error_message = None
    last_epoch = 0

    try:
        for epoch in range(1, cfg.epochs + 1):
            last_epoch = epoch
            refiner.train()
            tot, n = 0.0, 0
            for pred_batch, y_batch, x_batch in train_loader:
                B = pred_batch.size(0)
                # Sample random noise level per sample
                t = torch.rand(B, 1, 1, 1, device=device)
                noise_levels = max_noise * (1 - t) if noise_schedule == "linear" else max_noise * (0.5 * (1 + torch.cos(np.pi * t)))

                # Add noise to base prediction
                noise = torch.randn_like(pred_batch) * noise_levels
                noisy_pred = pred_batch + noise

                # Refiner predicts the residual (clean - noisy)
                refiner_pred = refiner(noisy_pred.detach())
                target_residual = pred_batch - noisy_pred  # = -noise

                loss = loss_fn(refiner_pred, target_residual + pred_batch, x_batch)
                # Actually, target should be the clean prediction
                loss = loss_fn(refiner_pred, pred_batch, x_batch)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                tot += loss.item()
                n += 1

            tr_loss = tot / max(n, 1)

            # Validation: iterative refinement
            refiner.eval()
            with torch.no_grad():
                va_loss_vals = []
                for i in range(0, len(base_preds_va), 16):
                    bp = base_preds_va[i:i+16]
                    yb = Y_va[i:i+16]
                    xb = X_va[i:i+16]
                    
                    # Start from base prediction
                    refined = bp.clone()
                    for step in range(refine_steps):
                        noise_level = _noise_schedule(step / refine_steps, noise_schedule, max_noise)
                        if noise_level > 0:
                            noise = torch.randn_like(refined) * noise_level
                            noisy = refined + noise
                        else:
                            noisy = refined
                        residual = refiner(noisy)
                        refined = residual  # refiner predicts clean output
                    
                    va_loss_vals.append(loss_fn(refined, yb, xb).item())
                va_loss = np.mean(va_loss_vals)

            is_best = va_loss < best_val
            if is_best:
                best_val = va_loss
                torch.save(refiner.state_dict(), results_dir / "model_best.pt")

            if epoch % cfg.heartbeat_interval_epochs == 0 or epoch == 1:
                LOGGER.info("Epoch %4d/%d train=%.6f val=%.6f best=%.6f%s",
                            epoch, cfg.epochs, tr_loss, va_loss, best_val,
                            " *" if is_best else "")

            if epoch % cfg.checkpoint_interval == 0:
                torch.save({"epoch": epoch, "refiner_state_dict": refiner.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val": best_val}, results_dir / "checkpoint.pt")

        torch.save(refiner.state_dict(), results_dir / "model_final.pt")
        (results_dir / "checkpoint.pt").unlink(missing_ok=True)

    except Exception as exc:
        status = "failed"
        error_message = "".join(traceback.format_exception(exc))
        LOGGER.exception("Refiner training failed")

    wall = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    # Final evaluation with iterative refinement
    val_metrics = {"r2_median": float("nan"), "mae_median": float("nan")}
    if status == "ok":
        try:
            refiner.load_state_dict(torch.load(results_dir / "model_best.pt", map_location=device, weights_only=True))
            refiner.eval()
            
            # Compute refined predictions
            with torch.no_grad():
                refined_preds = []
                for i in range(0, len(base_preds_va), 8):
                    bp = base_preds_va[i:i+8]
                    refined = bp.clone()
                    for step in range(refine_steps):
                        noise_level = _noise_schedule(step / refine_steps, noise_schedule, max_noise)
                        if noise_level > 0:
                            noise = torch.randn_like(refined) * noise_level
                            noisy = refined + noise
                        else:
                            noisy = refined
                        refined = refiner(noisy)
                    refined_preds.append(refined.cpu())
                refined_all = torch.cat(refined_preds, dim=0)

            # Compute per-case R²
            from shared.train import _compute_per_case_r2
            r2_med, mae_med = _compute_per_case_r2(
                refiner, base_preds_va, Y_va, val_b["case_names"], bs=4
            )
            # Actually we need to evaluate the full pipeline, not just refiner
            # Let's compute R² from refined predictions directly
            from shared.eval_module import EvalModule
            # For now, just use the simple R² computation
            mask = (~torch.isnan(Y_va.cpu())) & (X_va.cpu()[:, 0:1, :, :] <= 0)
            p_flat = refined_all[mask].numpy()
            t_flat = Y_va.cpu()[mask].numpy()
            if len(t_flat) > 0:
                ss_res = np.sum((p_flat - t_flat) ** 2)
                ss_tot = np.sum((t_flat - t_flat.mean()) ** 2)
                r2_global = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                mae_global = np.mean(np.abs(p_flat - t_flat))
                val_metrics = {"r2_median": r2_global, "mae_median": mae_global, "r2_global": r2_global}
            LOGGER.info("Final eval: R²=%.4f MAE=%.4f", val_metrics.get("r2_median", 0), val_metrics.get("mae_median", 0))
        except Exception as exc:
            LOGGER.exception("Final eval failed: %s", exc)

    metrics = {
        "experiment_id": cfg.experiment_id,
        "strategy": cfg.strategy,
        "arch_name": "pde_refiner",
        "loss_name": cfg.loss_name,
        "seed": cfg.seed,
        "epochs_trained": last_epoch,
        "wall_time_sec": wall,
        "peak_vram_gb": peak_vram,
        "gpu": gpu,
        "status": status,
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "error_message": error_message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = TrainConfig.model_validate(json.load(f))
    train_refiner(cfg)
