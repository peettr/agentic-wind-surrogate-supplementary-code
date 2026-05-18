"""Hash-locked raw-domain evaluator.

.. warning::

   **THIS MODULE IS LOCKED.** Its source is SHA-256-hashed into every
   ``metrics.json`` as ``eval_hash``. Any mutation invalidates the in-flight
   campaign. The locked behavior mirrors auto_v2's ``eval_seeds_v3.py``.

Evaluation contract (Appendix A.3 of framework doc):

* Domain      : raw wind speed via ``DataFormatterFixed.restore_raw_output_data``.
* Primary     : per-case RÂ² median.
* RÂ²          : ``1 - SS_res / SS_tot`` on pixels where ``truth >= 0 & finite``.
* Global RÂ²   : concatenation of all valid pixels across cases.
* Patch merge : ``np.nanmean(restored_patches, axis=0)``.
* Integrity   : before every ``evaluate()``, SHA-256 of this file AND of
                ``split_manifest.json`` are re-verified against the cached values.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_eval_hash() -> str:
    """SHA-256 of this module's source code â€” the eval fingerprint."""
    return _sha256_file(Path(__file__))


def compute_split_hash(split_manifest_path: str | Path) -> str:
    return _sha256_file(Path(split_manifest_path))


# ---------------------------------------------------------------------------
# DataFormatterFixed import (late; CRC-side path is preferred)
# ---------------------------------------------------------------------------
def _import_data_formatter() -> Any:
    """Import ``DataFormatterFixed`` from the v2 references tree.

    v2 eval uses DataFormatterFixed for restore_raw_output_data.
    The formatter lives on CRC at the v2 references directory.
    """
    candidates = [
        Path("<PROJECT_HPC_ROOT>/auto_v2/full_dataset/references"),
        Path("<PROJECT_HPC_ROOT>/auto_v2/full_dataset/scripts"),
        Path(__file__).resolve().parent.parent / "references",
    ]
    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        mod = importlib.import_module("data_formatter_fixed")
    except ImportError as e:
        raise ImportError(
            "DataFormatterFixed not found. Ensure the v2 references directory is "
            "on sys.path (see problem_definition.yaml)."
        ) from e
    return mod.DataFormatterFixed


# ---------------------------------------------------------------------------
# Per-case RÂ² on raw domain
# ---------------------------------------------------------------------------
def _per_case_r2(
    case_name: str,
    combined: np.ndarray,
    wind_angle: int,
    patch_indices: list[int],
    pred_all: np.ndarray,
    formatter_cls: Any,
    fmt_shape: int = 640,
) -> tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Restore ``patch_indices`` into raw domain and compute RÂ² + MAE."""
    formatter = formatter_cls(
        raw_data=[combined], wind_angles=[wind_angle], formatted_shape=fmt_shape
    )
    expected_n = formatter._fmt_input_data.shape[0]
    if len(patch_indices) != expected_n:
        return float("nan"), float("nan"), None, None

    pred_patches = np.stack([pred_all[i] for i in patch_indices], axis=0)
    pred_input = pred_patches[:, np.newaxis, :, :]                # (N, 1, fmt, fmt)
    restored = formatter.restore_raw_output_data(pred_input)      # (N, H, W)
    pred_raw = np.nanmean(restored, axis=0)
    truth = combined

    if truth.shape != pred_raw.shape:
        if truth.shape == pred_raw.T.shape:
            pred_raw = pred_raw.T
        else:
            return float("nan"), float("nan"), None, None

    valid = (truth >= 0) & np.isfinite(truth) & np.isfinite(pred_raw)
    t = truth[valid]
    p = pred_raw[valid]
    if t.size == 0:
        return float("nan"), float("nan"), None, None
    mae = float(np.mean(np.abs(t - p)))
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, mae, t, p


# ---------------------------------------------------------------------------
# EvalModule
# ---------------------------------------------------------------------------
class EvalModule:
    """Raw-domain evaluator with hash integrity enforcement."""

    def __init__(
        self,
        split_manifest_path: str | Path,
        data_dir: str | Path,
        fmt_shape: int = 640,
    ) -> None:
        self.split_manifest_path = Path(split_manifest_path)
        self.data_dir = Path(data_dir)
        self.fmt_shape = fmt_shape
        self.eval_hash = compute_eval_hash()
        self.split_hash = compute_split_hash(self.split_manifest_path)
        with open(self.split_manifest_path) as fh:
            self.manifest = json.load(fh)
        self._formatter_cls: Any = None  # lazy

    # --- integrity -------------------------------------------------------
    def verify_integrity(self) -> bool:
        """Re-verify both hashes; raise on drift."""
        now_eval = compute_eval_hash()
        now_split = compute_split_hash(self.split_manifest_path)
        if now_eval != self.eval_hash:
            raise RuntimeError(
                f"eval_module.py hash drift: stored={self.eval_hash[:12]}, "
                f"now={now_eval[:12]}"
            )
        if now_split != self.split_hash:
            raise RuntimeError(
                f"split_manifest.json hash drift: stored={self.split_hash[:12]}, "
                f"now={now_split[:12]}"
            )
        return True

    # --- helpers ---------------------------------------------------------
    def _load_formatter(self) -> Any:
        if self._formatter_cls is None:
            self._formatter_cls = _import_data_formatter()
        return self._formatter_cls

    def _load_split(self, split: str, seed: int = 1) -> dict:
        """Load split data from all_data.pt using split_manifest."""
        all_path = self.data_dir / "all_data.pt"
        if not all_path.exists():
            raise FileNotFoundError(all_path)
        bundle = torch.load(all_path, map_location="cpu", weights_only=False)

        # Get case names for this split
        seed_key = str(seed)
        if seed_key in self.manifest.get("seeds", {}):
            target_cases = set(self.manifest["seeds"][seed_key][split])
        else:
            raise ValueError(f"Seed {seed} not in split_manifest")

        p2c = bundle["patch_to_case"]
        case_names = bundle["case_names"]
        indices = [i for i in range(len(p2c)) if case_names[int(p2c[i])] in target_cases]
        idx_t = torch.tensor(indices, dtype=torch.long)

        return {
            "X": bundle["X"][idx_t],
            "Y": bundle["Y"][idx_t],
            "case_names": case_names,
            "patch_to_case": p2c[idx_t] if len(indices) > 0 else torch.tensor([], dtype=torch.long),
            "raw_cases": bundle.get("raw_cases", {}),
        }

    def _predict(
        self,
        model: torch.nn.Module,
        X: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> np.ndarray:
        model.eval()
        loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=False)
        preds: list[torch.Tensor] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu())
        out = torch.cat(preds, dim=0).numpy()[:, 0, :, :]
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    # --- public entry point ---------------------------------------------
    def evaluate(
        self,
        model: torch.nn.Module,
        split: str = "val",
        seed: int = 1,
        raw_cases: Optional[dict[str, tuple[np.ndarray, int]]] = None,
        batch_size: int = 4,
        device: Optional[torch.device] = None,
    ) -> dict[str, Any]:
        """Return metrics dict conforming to :class:`SplitMetrics`."""
        self.verify_integrity()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        bundle = self._load_split(split, seed=seed)
        X = bundle["X"]
        case_names: list[str] = bundle["case_names"]
        patch_to_case: torch.Tensor = bundle["patch_to_case"]

        if raw_cases is None:
            raw_cases = bundle.get("raw_cases", {})
            if not raw_cases:
                raise ValueError(
                    "No raw_cases in split bundle; pass raw_cases=... to evaluate()."
                )

        pred_all = self._predict(model, X, batch_size, device)
        formatter_cls = self._load_formatter()

        case_to_patches: dict[str, list[int]] = {}
        for i, ci in enumerate(patch_to_case.tolist()):
            case_to_patches.setdefault(case_names[ci], []).append(i)

        per_case_r2: dict[str, float] = {}
        per_case_mae: dict[str, float] = {}
        all_t_parts: list[np.ndarray] = []
        all_p_parts: list[np.ndarray] = []
        n_fail = 0
        for cn, patches in case_to_patches.items():
            if cn not in raw_cases:
                n_fail += 1
                continue
            combined, angle = raw_cases[cn]
            r2, mae, t, p = _per_case_r2(
                cn, combined, angle, patches, pred_all, formatter_cls, self.fmt_shape
            )
            if np.isnan(r2) or t is None:
                n_fail += 1
                continue
            per_case_r2[cn] = r2
            per_case_mae[cn] = mae
            all_t_parts.append(t)
            all_p_parts.append(p)

        r2_values = np.array(list(per_case_r2.values()), dtype=np.float64)
        mae_values = np.array(list(per_case_mae.values()), dtype=np.float64)

        if all_t_parts:
            all_t = np.concatenate(all_t_parts)
            all_p = np.concatenate(all_p_parts)
            ss_res = float(np.sum((all_t - all_p) ** 2))
            ss_tot = float(np.sum((all_t - all_t.mean()) ** 2))
            r2_global = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            r2_global = float("nan")

        return {
            "r2_median": float(np.median(r2_values)) if r2_values.size else float("nan"),
            "r2_mean": float(np.mean(r2_values)) if r2_values.size else float("nan"),
            "r2_global": float(r2_global),
            "mae_median": float(np.median(mae_values)) if mae_values.size else float("nan"),
            "mae_mean": float(np.mean(mae_values)) if mae_values.size else float("nan"),
            "per_case_r2": per_case_r2,
            "n_cases_evaluated": int(r2_values.size),
            "n_fail": int(n_fail),
        }


__all__ = ["EvalModule", "compute_eval_hash", "compute_split_hash"]



