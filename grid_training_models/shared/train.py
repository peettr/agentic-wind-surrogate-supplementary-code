"""Generic training loop driven by :class:`TrainConfig`.

Accepts any :class:`BaseSurrogate` registered with :data:`REGISTRY` and any
loss in :data:`LIBRARY`. Mirrors the locked training contract (Appendix A.2):

    * Adam, lr=1e-3, no scheduler
    * batch_size=16 (fallback 8 on OOM); ``zero_grad(set_to_none=True)`` after probe
    * Checkpoint every 50 epochs to ``checkpoint.pt``
    * ``model_best.pt`` on val-loss improvement; ``model_final.pt`` at end
    * Writes ``metrics.json`` (canonical schema) on completion
    * Condor sentinels: STARTED, HEARTBEAT.json, FINISHED / FAILED
    * Eviction recovery: resume from ``checkpoint.pt`` on restart
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import random
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Allow sibling imports when run as a standalone script (Condor).
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from shared.configs.schema import TrainConfig                                   # noqa: E402
from shared.eval_module import EvalModule                                       # noqa: E402
from shared.losses import LIBRARY as LOSS_LIBRARY                               # noqa: E402
from shared.models import REGISTRY as MODEL_REGISTRY                            # noqa: E402


LOGGER = logging.getLogger("auto_v3.train")


# Default training hyperparameters (v2 baseline values, but explorable in search).
DEFAULT_LR = 1.0e-3
DEFAULT_BATCH = 16
DEFAULT_BATCH_FALLBACK = 8


# ---------------------------------------------------------------------------
# Dynamic module loader for generated architectures / losses
# ---------------------------------------------------------------------------
def _load_external_module(script_path: str | Path):
    """Import a Python file at an absolute path as a fresh module.

    Used when ``cfg.script_path`` points at a codegen-generated architecture
    or loss that is not in the built-in :data:`MODEL_REGISTRY` / loss library.
    """
    p = Path(script_path)
    if not p.is_file():
        raise FileNotFoundError(f"script_path not found: {p}")
    mod_name = f"auto_v3_ext_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_model_cls(cfg: TrainConfig):
    """Resolve the model class: built-in first, then dynamic script_path."""
    if cfg.arch_name in MODEL_REGISTRY:
        return MODEL_REGISTRY.get(cfg.arch_name)
    if cfg.script_path:
        module = _load_external_module(cfg.script_path)
        # Generated model files use a canonical Model class. Prefer it before
        # scanning module globals, because wrapper files often import backbone
        # classes such as DilatedUNet; blindly taking the first nn.Module class
        # can bypass the wrapper and instantiate the backbone with unsafe
        # default widths.
        if hasattr(module, cfg.arch_name):
            return getattr(module, cfg.arch_name)
        if hasattr(module, "Model"):
            model_attr = getattr(module, "Model")
            if isinstance(model_attr, type) and issubclass(model_attr, torch.nn.Module):
                return model_attr
        for name, obj in module.__dict__.items():
            if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                continue
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                return obj
        raise AttributeError(
            f"No torch.nn.Module class named {cfg.arch_name!r} or 'Model' in {cfg.script_path}"
        )
    raise KeyError(
        f"Unknown arch_name {cfg.arch_name!r} and no script_path provided."
    )


def _resolve_loss(cfg: TrainConfig) -> torch.nn.Module:
    """Resolve the loss: built-in library first, then dynamic script_path."""
    try:
        return LOSS_LIBRARY.build(cfg.loss_name, **cfg.loss_kwargs)
    except KeyError:
        if not cfg.script_path:
            raise
        module = _load_external_module(cfg.script_path)
        if hasattr(module, cfg.loss_name):
            return getattr(module, cfg.loss_name)(**cfg.loss_kwargs)
        raise


# ---------------------------------------------------------------------------
# Locked-contract enforcement
# ---------------------------------------------------------------------------
def _enforce_locked_contract(cfg: TrainConfig) -> TrainConfig:
    """Only enforce truly locked parameters (data pipeline, eval domain).

    Training hyperparams (lr, batch_size, scheduler) are NOT locked —
    the search strategies may explore them. We only cap batch_size as a
    safety guard against OOM before the first iteration.
    """
    changed: dict = {}
    # Safety: cap batch_size to avoid instant OOM.
    if cfg.batch_size > 16:
        LOGGER.warning(
            "batch_size=%d capped to 16", cfg.batch_size,
        )
        changed["batch_size"] = 16
    if changed:
        cfg = cfg.model_copy(update=changed)
    return cfg


# ---------------------------------------------------------------------------
# Sentinel files
# ---------------------------------------------------------------------------
def _write_sentinel(results_dir: Path, name: str, payload: Optional[dict] = None) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    p = results_dir / name
    if payload is None:
        p.write_text(datetime.utcnow().isoformat() + "Z\n")
    else:
        p.write_text(json.dumps(payload, indent=2))


def _write_failure_metrics(
    results_dir: Path,
    cfg: TrainConfig,
    *,
    status: str,
    error: str,
    wall_time: float,
    peak_vram: float,
    gpu: str,
    epochs_trained: int = 0,
) -> dict:
    """Write canonical metrics.json for early failures before eval is possible."""
    metrics = {
        "experiment_id": cfg.experiment_id,
        "strategy": os.environ.get("AUTO_V3_STRATEGY") or cfg.strategy or "unknown",
        "arch_name": cfg.arch_name,
        "loss_name": cfg.loss_name,
        "arch_kwargs": dict(cfg.arch_kwargs or {}),
        "loss_kwargs": dict(cfg.loss_kwargs or {}),
        "seed": cfg.seed,
        "epochs_trained": int(epochs_trained),
        "wall_time_sec": float(wall_time),
        "peak_vram_gb": float(peak_vram),
        "gpu": gpu,
        "status": status,
        "val_metrics": None,
        "holdout_metrics": None,
        "config_hash": _config_hash(cfg),
        "eval_hash": "",
        "split_hash": "",
        "error_message": error,
        "early_stop_info": None,
        "script_path": cfg.script_path,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------
def _config_hash(cfg: TrainConfig) -> str:
    payload = cfg.model_dump()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_all_data(data_dir: Path) -> dict:
    """Load consolidated all_data.pt (all cases, all patches)."""
    return torch.load(data_dir / "all_data.pt", map_location="cpu", weights_only=False)


def _load_manifest(manifest_path: Path) -> dict:
    return json.loads(Path(manifest_path).read_text())


def _split_patches_by_cases(
    bundle: dict,
    target_cases: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select patches whose case is in target_cases. Returns (X, Y)."""
    p2c = bundle["patch_to_case"]
    case_names = bundle["case_names"]
    indices = [i for i in range(len(p2c)) if case_names[int(p2c[i])] in target_cases]
    indices_t = torch.tensor(indices, dtype=torch.long)
    return bundle["X"][indices_t], bundle["Y"][indices_t]


def _load_train_val(
    data_dir: Path,
    manifest_path: str,
    seed: int,
) -> tuple[dict, dict, dict]:
    """Load all_data.pt, split by seed's manifest into train/val/holdout dicts.

    Each returned dict has X, Y, case_names (subset only).
    """
    bundle = _load_all_data(data_dir)
    manifest = _load_manifest(Path(manifest_path))
    seed_key = str(seed)
    if seed_key not in manifest["seeds"]:
        raise ValueError(f"Seed {seed} not in split_manifest (available: {list(manifest['seeds'].keys())})")
    sp = manifest["seeds"][seed_key]

    train_set = set(sp["train"])
    val_set = set(sp["val"])
    holdout_set = set(sp["holdout"])

    p2c = bundle["patch_to_case"]
    case_names = bundle["case_names"]

    def _extract(target_set: set[str]) -> dict:
        indices = [i for i in range(len(p2c)) if case_names[int(p2c[i])] in target_set]
        idx_t = torch.tensor(indices, dtype=torch.long)
        return {
            "X": bundle["X"][idx_t],
            "Y": bundle["Y"][idx_t],
            "case_names": [case_names[int(p2c[i])] for i in indices],
        }

    return _extract(train_set), _extract(val_set), _extract(holdout_set)


# ---------------------------------------------------------------------------
# SDF / normal feature computation
# ---------------------------------------------------------------------------
def _compute_sdf_features(X: torch.Tensor, mode: str) -> torch.Tensor:
    """Compute SDF and/or normal channels from building height input.
    
    Args:
        X: (N, 1, H, W) building height tensor (0 outside buildings)
        mode: "height_sdf" or "height_sdf_normal"
    
    Returns:
        (N, C, H, W) where C=2 (height_sdf) or C=3 (height_sdf_normal)
    """
    import scipy.ndimage as ndi
    
    N, _, H, W = X.shape
    height = X[:, 0, :, :].cpu().numpy()  # (N, H, W)
    building_mask = height > 0  # (N, H, W) bool
    
    sdf_list = []
    normal_list = []
    
    for i in range(N):
        mask = building_mask[i]  # (H, W)
        # Signed distance: negative inside, positive outside
        dist_outside = ndi.distance_transform_edt(~mask)
        dist_inside = ndi.distance_transform_edt(mask)
        sdf = dist_outside - dist_inside  # (H, W)
        sdf_list.append(torch.from_numpy(sdf).float())
        
        if mode == "height_sdf_normal":
            # Normal = gradient of SDF, normalized
            gy, gx = np.gradient(sdf)
            mag = np.sqrt(gx**2 + gy**2 + 1e-8)
            # Store as angle: atan2(gy, gx) -> single channel
            angle = np.arctan2(gy, gx)
            normal_list.append(torch.from_numpy(angle).float())
    
    sdf_batch = torch.stack(sdf_list, dim=0).unsqueeze(1)  # (N, 1, H, W)
    
    if mode == "height_sdf":
        return torch.cat([X, sdf_batch.to(X.device)], dim=1)  # (N, 2, H, W)
    else:
        normal_batch = torch.stack(normal_list, dim=0).unsqueeze(1)  # (N, 1, H, W)
        return torch.cat([X, sdf_batch.to(X.device), normal_batch.to(X.device)], dim=1)  # (N, 3, H, W)


# ---------------------------------------------------------------------------
# OOM probe & eval helpers
# ---------------------------------------------------------------------------
def _probe_batch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
) -> int:
    """Probe whether *batch_size* fits in VRAM. Returns usable batch size or 0 on failure."""
    try:
        p = model(X[:batch_size])
        loss = loss_fn(p, Y[:batch_size], X[:batch_size])
        loss.backward()
        optimizer.zero_grad(set_to_none=True)  # clear probe gradients (Appendix A.2 #20)
        del p, loss
        torch.cuda.empty_cache()
        LOGGER.info("Batch %d OK", batch_size)
        return batch_size
    except RuntimeError as exc:
        torch.cuda.empty_cache()
        if batch_size > 8:
            LOGGER.warning("OOM with batch=%d (%s); falling back to 8", batch_size, exc)
            return _probe_batch(model, loss_fn, X, Y, optimizer, 8)
        else:
            LOGGER.error("OOM even with batch=8 (%s); aborting this experiment.", exc)
            return 0


def _eval_split(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    bs: int,
) -> float:
    model.eval()
    with torch.no_grad():
        preds = [model(X[i : i + bs]) for i in range(0, len(X), bs)]
        pred = torch.cat(preds, dim=0)
        return loss_fn(pred, Y, X).item()


def _compute_per_case_r2(
    model: torch.nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    case_names: list[str],
    bs: int = 4,
) -> tuple[float, float]:
    """Return (r2_median, mae_median) over per-case R² on valid (non-NaN, non-building) pixels."""
    import numpy as np

    model.eval()
    case_indices: dict[str, list[int]] = {}
    for i, cn in enumerate(case_names):
        case_indices.setdefault(cn, []).append(i)

    r2s, maes = [], []
    with torch.no_grad():
        for cn in sorted(case_indices):
            idx = case_indices[cn]
            x = X[idx]
            y = Y[idx]
            pred = torch.cat([model(x[i : i + 1]) for i in range(len(x))], dim=0)
            mask = (~torch.isnan(y)) & (x[:, 0:1, :, :] <= 0)
            p_flat = pred[mask].cpu().numpy()
            t_flat = y[mask].cpu().numpy()
            if len(t_flat) < 10:
                continue
            ss_res = float(np.sum((p_flat - t_flat) ** 2))
            ss_tot = float(np.sum((t_flat - t_flat.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mae = float(np.mean(np.abs(p_flat - t_flat)))
            r2s.append(r2)
            maes.append(mae)

    if not r2s:
        return 0.0, 0.0
    return float(np.median(r2s)), float(np.median(maes))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig) -> dict:
    """Run end-to-end training; return the metrics dict written to metrics.json."""
    cfg = _enforce_locked_contract(cfg)
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

    _write_sentinel(results_dir, "STARTED")
    t0 = time.time()
    set_deterministic(cfg.seed)

    # Extra training hyperparams from arch_kwargs.training (parse FIRST).
    # Keep these out of the model constructor because most architecture
    # classes do not accept a `training` keyword. The materialized V5 configs
    # use this namespace for scheduler/EMA/data_augment without expanding the
    # TrainConfig schema.
    model_kwargs = dict(cfg.arch_kwargs or {})
    training_extras = model_kwargs.pop("training", {}) or {}
    weight_decay = float(training_extras.get("weight_decay", 0))
    grad_clip = training_extras.get("grad_clip")  # e.g. 0.5 or None
    scheduler_name = training_extras.get("scheduler")  # e.g. "cosine"
    optimizer_name = str(training_extras.get("optimizer", "adam")).lower()
    dropout_rate = float(training_extras.get("dropout", 0))
    ema_decay = training_extras.get("ema_decay")  # e.g. 0.999 or None
    data_augment = bool(training_extras.get("data_augment", False))

    # Model + loss --------------------------------------------------------
    model_cls = _resolve_model_cls(cfg)
    model = model_cls(**model_kwargs)
    loss_fn = _resolve_loss(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    model = model.to(device)

    # Data ---------------------------------------------------------------
    train_b, val_b, holdout_b = _load_train_val(
        data_dir, cfg.split_manifest_path, cfg.seed,
    )
    X_tr, Y_tr = train_b["X"].to(device), train_b["Y"].to(device)
    X_va, Y_va = val_b["X"].to(device), val_b["Y"].to(device)
    val_cases = val_b["case_names"]
    # Use only Val for best-model selection (holdout locked for final evaluation)
    X_test, Y_test = X_va, Y_va
    assert not torch.isnan(X_tr).any(), "X_train contains NaN"
    assert not torch.isnan(X_test).any(), "X_val contains NaN"
    
    # SDF feature augmentation
    if cfg.input_features != "height":
        LOGGER.info("Computing SDF features: %s", cfg.input_features)
        X_tr = _compute_sdf_features(X_tr, cfg.input_features)
        X_test = _compute_sdf_features(X_test, cfg.input_features)
        X_va = _compute_sdf_features(X_va, cfg.input_features)
        LOGGER.info("Input shape after SDF: %s", tuple(X_tr.shape))
    
    LOGGER.info("Data: train=%d val=%d holdout=%d patches",
                len(X_tr), len(X_va), len(holdout_b["X"]))

    # EvalModule for official R² (if compute_r2 enabled)
    eval_mod = None
    if cfg.compute_r2:
        eval_mod = EvalModule(
            split_manifest_path=cfg.split_manifest_path,
            data_dir=str(data_dir),
        )
        LOGGER.info("Official R² evaluation enabled (eval_module.py)")
        
        # Monkey-patch eval_mod's _predict for SDF features (heartbeat path)
        if cfg.input_features != "height":
            _orig_predict_heartbeat = eval_mod._predict
            def _predict_with_sdf_heartbeat(model, X, batch_size, device):
                X_aug = _compute_sdf_features(X, cfg.input_features)
                return _orig_predict_heartbeat(model, X_aug, batch_size, device)
            eval_mod._predict = _predict_with_sdf_heartbeat

    LOGGER.info(
        "Device=%s GPU=%s params=%d",
        device, gpu, sum(p.numel() for p in model.parameters()),
    )

    # EMA (Exponential Moving Average) setup
    ema_model = None
    if ema_decay is not None:
        from copy import deepcopy
        ema_model = deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        LOGGER.info("EMA enabled with decay=%.4f", ema_decay)

    # Data augmentation
    if data_augment:
        class AugmentedDataset(torch.utils.data.Dataset):
            def __init__(self, X, Y):
                self.X, self.Y = X, Y
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                x, y = self.X[idx], self.Y[idx]
                if torch.rand(1) < 0.5:
                    x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
                if torch.rand(1) < 0.5:
                    x, y = torch.flip(x, [-2]), torch.flip(y, [-2])
                return x, y
        train_dataset = AugmentedDataset(X_tr, Y_tr)
    else:
        train_dataset = TensorDataset(X_tr, Y_tr)

    # activation & norm_type are model-level, passed via arch_kwargs

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9,
            weight_decay=weight_decay,
        )
    else:  # default: adam
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=weight_decay,
        )

    # Optional scheduler
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01,
        )
    elif scheduler_name == "warmup":
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=50)
        main = CosineAnnealingLR(optimizer, T_max=cfg.epochs - 50)
        scheduler = SequentialLR(optimizer, [warmup, main], milestones=[50])

    # Resume from checkpoint if present (eviction recovery) --------------
    ckpt_path = results_dir / "checkpoint.pt"
    start_epoch = 1
    best_val = float("inf")
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val = ck["best_val"]
        LOGGER.info("Resumed from epoch %d, best_val=%.6f", ck["epoch"], best_val)
        # Fix #4: restore EMA state from checkpoint
        if ema_model is not None and ck.get("ema_state_dict") is not None:
            ema_model.load_state_dict(ck["ema_state_dict"])
            LOGGER.info("Restored EMA state from checkpoint")
        if "best_r2_so_far" in ck:
            best_r2_so_far = ck["best_r2_so_far"]
            LOGGER.info("Restored best_r2_so_far=%.4f from checkpoint", best_r2_so_far)

    bs = cfg.batch_size
    if device.type == "cuda":
        bs = _probe_batch(model, loss_fn, X_tr, Y_tr, optimizer, bs)
        if bs == 0:
            # Model too large even at batch=8 — abort.
            LOGGER.error("Model too large for GPU; writing FAILED sentinel.")
            _write_sentinel(results_dir, "FAILED", {"error": "OOM at batch=8"})
            metrics = _write_failure_metrics(
                results_dir,
                cfg,
                status="failed",
                error="OOM at batch=8",
                wall_time=time.time() - t0,
                peak_vram=0,
                gpu=gpu,
            )
            return metrics
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    EVAL_BS = 16

    peak_vram = 0.0
    status = "ok"
    error_message: Optional[str] = None
    last_epoch = start_epoch - 1
    # best_r2_so_far may be set by checkpoint restore above; only reset if not restored
    try:
        best_r2_so_far
    except NameError:
        best_r2_so_far = -float('inf')  # track best R² for early stop comparison
    early_stop_info = None  # store early stop reason

    # Load baseline R² curve for early stop comparison (once, before loop)
    baseline_r2_curve = None
    baseline_curve_path = getattr(cfg, 'baseline_r2_curve_path', None)
    if baseline_curve_path:
        try:
            with open(baseline_curve_path) as _f:
                baseline_r2_curve = json.load(_f)
            if not baseline_r2_curve:
                baseline_r2_curve = None
                LOGGER.warning("Baseline R² curve is empty, early stop disabled")
            else:
                LOGGER.info("Loaded baseline R² curve from %s (%d points)", baseline_curve_path, len(baseline_r2_curve))
        except Exception as _e:
            LOGGER.warning("Failed to load baseline R² curve: %s", _e)
    early_stop_min = getattr(cfg, 'early_stop_wall_min', 100)
    max_wall_min = getattr(cfg, 'max_wall_min', 200)
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            last_epoch = epoch
            model.train()
            tot, n = 0.0, 0
            for xb, yb in train_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb, xb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip,
                    )
                optimizer.step()
                if ema_model is not None:
                    with torch.no_grad():
                        for ep, mp in zip(ema_model.parameters(), model.parameters()):
                            ep.mul_(ema_decay).add_(mp, alpha=1 - ema_decay)
                        # Fix #1: copy BatchNorm buffers (running_mean/var)
                        # Buffers are already EMAs maintained by BN, so copy directly
                        for eb, mb in zip(ema_model.buffers(), model.buffers()):
                            eb.copy_(mb)
                tot += loss.item()
                n += 1
            tr_loss = tot / max(n, 1)
            eval_model = ema_model if ema_model is not None else model
            va_loss = _eval_split(eval_model, loss_fn, X_test, Y_test, EVAL_BS)
            if scheduler is not None:
                scheduler.step()

            is_best = va_loss < best_val
            if is_best:
                best_val = va_loss
                torch.save(eval_model.state_dict(), results_dir / "model_best.pt")

            if device.type == "cuda":
                peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)

            if epoch % cfg.heartbeat_interval_epochs == 0 or epoch == 1:
                r2_med, mae_med = 0.0, 0.0
                if cfg.compute_r2:
                    eval_model_r2 = ema_model if ema_model is not None else model
                    if eval_mod is not None:
                        r2_result = eval_mod.evaluate(
                            eval_model_r2, split="val", device=device, seed=cfg.seed,
                        )
                        r2_med = r2_result["r2_median"]
                        mae_med = r2_result["mae_median"]
                    else:
                        r2_med, mae_med = _compute_per_case_r2(
                            eval_model_r2, X_test, Y_test, val_cases, bs=4,
                        )
                _write_sentinel(
                    results_dir,
                    "HEARTBEAT.json",
                    {
                        "epoch": epoch,
                        "train_loss": tr_loss,
                        "val_loss": va_loss,
                        "best_val": best_val,
                        "r2_median": r2_med if cfg.compute_r2 else None,
                        "mae_median": mae_med if cfg.compute_r2 else None,
                        "peak_vram_gb": peak_vram,
                        "time": datetime.utcnow().isoformat() + "Z",
                    },
                )
                if cfg.compute_r2 and r2_med > best_r2_so_far:
                    best_r2_so_far = r2_med
                if cfg.compute_r2:
                    LOGGER.info(
                        "Epoch %4d/%d train=%.6f val=%.6f best=%.6f R²=%.4f MAE=%.4f%s",
                        epoch, cfg.epochs, tr_loss, va_loss, best_val, r2_med, mae_med,
                        " *" if is_best else "",
                    )
                else:
                    LOGGER.info(
                        "Epoch %4d/%d train=%.6f val=%.6f best=%.6f%s",
                        epoch, cfg.epochs, tr_loss, va_loss, best_val,
                        " *" if is_best else "",
                    )

            # Wall-time early stop: use baseline curve loaded before loop
            elapsed_min = (time.time() - t0) / 60.0
            stop_reason = None

            # Rule 1: 100min + best R² not beating baseline at same epoch → stop
            if elapsed_min >= early_stop_min and baseline_r2_curve is not None and cfg.compute_r2:
                # Compare against most recently passed baseline epoch (not future)
                baseline_epochs = sorted([int(k) for k in baseline_r2_curve.keys()])
                past_epochs = [e for e in baseline_epochs if e <= epoch]
                if past_epochs:
                    ref_epoch = past_epochs[-1]
                else:
                    ref_epoch = baseline_epochs[0]
                baseline_r2_at_epoch = baseline_r2_curve[str(ref_epoch)]
                if best_r2_so_far < baseline_r2_at_epoch:
                    stop_reason = f"early_stop: {elapsed_min:.0f}min, best_R²={best_r2_so_far:.4f} < baseline@ep{ref_epoch}={baseline_r2_at_epoch:.4f}"

            # Rule 2: 200min absolute limit → stop regardless
            if elapsed_min >= max_wall_min:
                stop_reason = f"max_wall_time: {elapsed_min:.0f}min >= {max_wall_min}min limit"

            if stop_reason:
                LOGGER.info("\u23f9 %s at epoch %d", stop_reason, epoch)
                final_model = ema_model if ema_model is not None else model
                torch.save(final_model.state_dict(), results_dir / "model_final.pt")
                if ckpt_path.exists():
                    ckpt_path.unlink(missing_ok=True)
                # Keep status="ok" so downstream eval/sentinel still runs
                status = "ok"
                early_stop_info = stop_reason  # store separately
                break

            if epoch % cfg.checkpoint_interval == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "ema_state_dict": ema_model.state_dict() if ema_model else None,
                        "best_val": best_val,
                        "best_r2_so_far": best_r2_so_far,
                        "seed": cfg.seed,
                    },
                    ckpt_path,
                )
                LOGGER.info("Checkpoint saved at epoch %d", epoch)

        final_model = ema_model if ema_model is not None else model
        torch.save(final_model.state_dict(), results_dir / "model_final.pt")
        if ckpt_path.exists():
            ckpt_path.unlink(missing_ok=True)
    except Exception as exc:  # runtime safety net — write FAILED sentinel
        status = "failed"
        error_message = "".join(traceback.format_exception(exc))
        _write_sentinel(results_dir, "FAILED", {"error": error_message})
        LOGGER.exception("Training failed")

    wall = time.time() - t0

    def _nan_split() -> dict:
        return {
            "r2_median": float("nan"), "r2_mean": float("nan"),
            "r2_global": float("nan"), "mae_median": float("nan"),
            "mae_mean": float("nan"), "per_case_r2": {},
        }

    # Raw-domain evaluation on val (+ holdout if requested) ---------------
    val_metrics = _nan_split()
    holdout_metrics: Optional[dict] = None
    eval_hash = split_hash = ""
    if status == "ok":
        try:
            evaluator = EvalModule(cfg.split_manifest_path, cfg.data_dir)
            eval_hash = evaluator.eval_hash
            split_hash = evaluator.split_hash
            best_state = torch.load(
                results_dir / "model_best.pt", map_location=device, weights_only=True
            )
            model.load_state_dict(best_state)
            splits = cfg.eval_splits or ["val"]
            
            # Wrap evaluator to inject SDF features if needed
            if cfg.input_features != "height":
                _orig_predict = evaluator._predict
                def _predict_with_sdf(model, X, batch_size, device):
                    X_aug = _compute_sdf_features(X, cfg.input_features)
                    return _orig_predict(model, X_aug, batch_size, device)
                evaluator._predict = _predict_with_sdf
            
            if "val" in splits:
                val_metrics = evaluator.evaluate(model, split="val", seed=cfg.seed, device=device)
            if "holdout" in splits:
                holdout_metrics = evaluator.evaluate(
                    model, split="holdout", seed=cfg.seed, device=device,
                )
        except Exception as exc:
            LOGGER.exception("Eval failed: %s", exc)
            status = "failed"
            error_message = str(exc)
            # Ensure sentinel is consistent on eval failure (fix: sentinels
            # were previously inconsistent when eval failed after a clean
            # training loop).
            _write_sentinel(results_dir, "FAILED", {"error": error_message})

    def _pack_split(m: Optional[dict]) -> Optional[dict]:
        if m is None:
            return None
        return {
            "r2_median": m.get("r2_median"),
            "r2_mean": m.get("r2_mean"),
            "r2_global": m.get("r2_global"),
            "mae_median": m.get("mae_median"),
            "mae_mean": m.get("mae_mean"),
            "per_case_r2": m.get("per_case_r2", {}),
        }

    strategy_env = os.environ.get("AUTO_V3_STRATEGY")
    strategy = strategy_env or cfg.strategy or "unknown"

    metrics = {
        "experiment_id": cfg.experiment_id,
        "strategy": strategy,
        "arch_name": cfg.arch_name,
        "loss_name": cfg.loss_name,
        "arch_kwargs": dict(cfg.arch_kwargs or {}),
        "loss_kwargs": dict(cfg.loss_kwargs or {}),
        "seed": cfg.seed,
        "epochs_trained": int(last_epoch),
        "wall_time_sec": wall,
        "peak_vram_gb": float(peak_vram),
        "gpu": gpu,
        "status": status,
        "val_metrics": _pack_split(val_metrics) or _pack_split(_nan_split()),
        "holdout_metrics": _pack_split(holdout_metrics),
        "config_hash": _config_hash(cfg),
        "eval_hash": eval_hash,
        "split_hash": split_hash,
        "error_message": error_message,
        "early_stop_info": early_stop_info,
        "script_path": cfg.script_path,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if status == "ok":
        _write_sentinel(results_dir, "FINISHED")
    return metrics


# ---------------------------------------------------------------------------
# Eval-only mode (holdout for final validation)
# ---------------------------------------------------------------------------
def eval_only(cfg: TrainConfig, model_path: str | Path, split: str = "holdout") -> dict:
    """Run EvalModule on a pre-trained model and return raw-domain metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = _resolve_model_cls(cfg)
    model = model_cls(**cfg.arch_kwargs).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    evaluator = EvalModule(cfg.split_manifest_path, cfg.data_dir)
    
    # Monkey-patch evaluator for SDF features if needed
    if cfg.input_features != "height":
        _orig_predict = evaluator._predict
        def _predict_with_sdf(model, X, batch_size, device):
            X_aug = _compute_sdf_features(X, cfg.input_features)
            return _orig_predict(model, X_aug, batch_size, device)
        evaluator._predict = _predict_with_sdf
    
    return evaluator.evaluate(model, split=split, seed=cfg.seed, device=device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> tuple[TrainConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="Path to TrainConfig JSON file")
    ap.add_argument(
        "--eval-holdout", action="store_true",
        help="Skip training; run only EvalModule on holdout using model_best.pt.",
    )
    ap.add_argument(
        "--model-path", type=str, default=None,
        help="Override model checkpoint path for --eval-holdout mode.",
    )
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = TrainConfig.model_validate(json.load(fh))
    return cfg, args


if __name__ == "__main__":
    _cfg, _args = _parse_args()
    if _args.eval_holdout:
        # Dedicated holdout-only evaluation path (fix #4): run EvalModule on a
        # trained checkpoint and emit metrics.json with holdout_metrics filled.
        model_path = (
            _args.model_path
            or str(Path(_cfg.results_dir) / "model_best.pt")
        )
        t0 = time.time()
        out = eval_only(_cfg, model_path, split="holdout")
        metrics = {
            "experiment_id": _cfg.experiment_id,
            "strategy": os.environ.get(
                "AUTO_V3_STRATEGY", _cfg.strategy or "unknown",
            ),
            "arch_name": _cfg.arch_name,
            "loss_name": _cfg.loss_name,
            "arch_kwargs": dict(_cfg.arch_kwargs or {}),
            "loss_kwargs": dict(_cfg.loss_kwargs or {}),
            "seed": _cfg.seed,
            "epochs_trained": 0,
            "wall_time_sec": time.time() - t0,
            "peak_vram_gb": 0.0,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "CPU"
            ),
            "status": "ok",
            "val_metrics": {
                "r2_median": float("nan"), "r2_mean": float("nan"),
                "r2_global": None, "mae_median": None,
                "mae_mean": None, "per_case_r2": {},
            },
            "holdout_metrics": {
                "r2_median": out.get("r2_median"),
                "r2_mean": out.get("r2_mean"),
                "r2_global": out.get("r2_global"),
                "mae_median": out.get("mae_median"),
                "mae_mean": out.get("mae_mean"),
                "per_case_r2": out.get("per_case_r2", {}),
            },
            "config_hash": _config_hash(_cfg),
            "eval_hash": "",
            "split_hash": "",
            "error_message": None,
            "script_path": _cfg.script_path,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        Path(_cfg.results_dir).mkdir(parents=True, exist_ok=True)
        (Path(_cfg.results_dir) / "metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
    else:
        train(_cfg)
