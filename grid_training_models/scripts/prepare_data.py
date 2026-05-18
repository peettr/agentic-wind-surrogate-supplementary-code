"""Step 0: raw UrbanTALES cases -> single consolidated dataset + multi-seed manifest.

Design (v2-learned, 2026-04-17):
  - Generate ONE file with ALL cases and ALL patches.
  - Split happens at train time, not here.
  - Lu's 20-seed CSVs define train/test per seed.
  - Test set further split into val + holdout (stratified, seed=42).

Outputs (into --out-dir):
  all_data.pt       — X, Y, nan_mask, case_names, wind_angles,
                       patch_to_case, fmt_shape, raw_cases
  split_manifest.json — 20 seeds × (train/val/holdout case lists) + hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import xarray as xr


DEFAULT_RAW_DIR = Path(
    "<PROJECT_HPC_ROOT>/data/urbantales/raw"
)
DEFAULT_FMT_PATHS = [
    Path("<PROJECT_HPC_ROOT>/auto_v2/full_dataset/references"),
]
DEFAULT_SPLIT_DIR = Path(
    "<PROJECT_HPC_ROOT>/auto_v2/full_dataset/references/all_cases_20exp"
)
DEFAULT_FMT_SHAPE = 640
VAL_HOLDOUT_SEED = 42


def _import_data_formatter():
    for p in DEFAULT_FMT_PATHS:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import data_formatter  # type: ignore
    return data_formatter.DataFormatter


def load_combined_with_nan(case_dir: Path, case_name: str) -> np.ndarray:
    topo = np.flipud(
        np.loadtxt(case_dir / f"{case_name}_topo", dtype=np.float32)
    )
    with xr.open_dataset(case_dir / f"{case_name}_ped.nc") as ds:
        uped = ds["Uped"].values.astype(np.float32)
    combined = uped.copy()
    combined[topo > 0] = -topo[topo > 0]
    return combined


def case_type(name: str) -> str:
    for t in ("VA", "VS", "UA", "US"):
        if name.startswith(t):
            return t
    return "REAL"


def load_lu_split_for_seed(split_dir: Path, seed: int) -> tuple[set[str], set[str]]:
    """Load Lu's train/test case names for a given seed."""
    train_path = split_dir / f"metrics_in_training_set_seed{seed}.csv"
    test_path = split_dir / f"metrics_in_test_set_seed{seed}.csv"

    def _parse(path: Path) -> set[str]:
        names: set[str] = set()
        for line in path.read_text().strip().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) >= 2 and parts[0] != "total":
                names.add(f"{parts[0]}_d{int(float(parts[1])):02d}")
        return names

    return _parse(train_path), _parse(test_path)


def stratify_test_split(
    test_names: list[str],
    case_data: dict[str, dict],
    val_frac: float = 0.5,
    seed: int = VAL_HOLDOUT_SEED,
) -> tuple[list[str], list[str]]:
    """Split a test set into val + holdout, stratified."""
    rng = random.Random(seed)
    n_density_bins, n_wind_bins = 3, 4

    # Compute density per case from raw combined
    densities = {}
    for n in test_names:
        if n in case_data:
            densities[n] = float((case_data[n]["combined"] < 0).mean())
        else:
            densities[n] = 0.0

    dsorted = sorted(densities.values())
    d_edges = [dsorted[i * len(dsorted) // n_density_bins]
               for i in range(1, n_density_bins)]
    wind_edges = np.linspace(0, 360, n_wind_bins + 1)[1:-1]

    def d_bin(d):
        for i, e in enumerate(d_edges):
            if d < e: return i
        return n_density_bins - 1

    def w_bin(name):
        m = re.search(r"_d(\d+)$", name)
        w = int(m.group(1)) if m else 0
        for i, e in enumerate(wind_edges):
            if w < e: return i
        return n_wind_bins - 1

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for n in test_names:
        key = (case_type(n), d_bin(densities[n]), w_bin(n))
        buckets[key].append(n)

    val, holdout = [], []
    for names in buckets.values():
        rng.shuffle(names)
        k = max(1, int(round(len(names) * val_frac)))
        val.extend(names[:k])
        holdout.extend(names[k:])
    return sorted(val), sorted(holdout)


def format_all_cases(
    cases: list[dict], formatter_cls, fmt_shape: int,
) -> dict:
    raw_data = [c["combined"] for c in cases]
    wind_angles = [c["wind_angle"] for c in cases]
    fmt = formatter_cls(
        raw_data=raw_data,
        wind_angles=wind_angles,
        formatted_shape=fmt_shape,
    )
    X_fmt = fmt.get_formatted_input_data().astype(np.float32)
    Y_fmt = fmt.get_formatted_output_data()
    nan_mask = np.isnan(Y_fmt)
    X_clean = np.nan_to_num(X_fmt, nan=0.0).astype(np.float32)

    patch_to_case: list[int] = []
    for ci, sr in enumerate(fmt._slice_indices):
        patch_to_case.extend([ci] * len(sr))

    return {
        "X": torch.from_numpy(X_clean).float(),
        "Y": torch.from_numpy(Y_fmt).float(),
        "nan_mask": torch.from_numpy(nan_mask).bool(),
        "case_names": [c["name"] for c in cases],
        "wind_angles": wind_angles,
        "patch_to_case": torch.tensor(patch_to_case, dtype=torch.long),
        "fmt_shape": fmt_shape,
        "raw_cases": {
            c["name"]: (c["combined"], c["wind_angle"]) for c in cases
        },
    }


def sha256_of(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode()
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--fmt-shape", type=int, default=DEFAULT_FMT_SHAPE)
    ap.add_argument("--val-frac", type=float, default=0.5)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    formatter_cls = _import_data_formatter()

    # --- Load all raw cases ---
    all_cases: dict[str, dict] = {}
    errors: list[str] = []
    for d in sorted(args.raw_dir.iterdir()):
        if not d.is_dir():
            continue
        m = re.search(r"_d(\d+)$", d.name)
        if not m:
            continue
        try:
            combined = load_combined_with_nan(d, d.name)
            all_cases[d.name] = {
                "name": d.name,
                "combined": combined,
                "wind_angle": int(m.group(1)),
            }
        except Exception as exc:
            errors.append(f"{d.name}: {exc}")
    print(f"Loaded {len(all_cases)} raw cases ({len(errors)} errors)")

    # --- Format ALL cases into one bundle ---
    cases_sorted = [all_cases[n] for n in sorted(all_cases.keys())]
    print(f"Formatting {len(cases_sorted)} cases (fmt_shape={args.fmt_shape})...")
    bundle = format_all_cases(cases_sorted, formatter_cls, args.fmt_shape)
    torch.save(bundle, out_dir / "all_data.pt")
    print(f"  wrote all_data.pt: X={bundle['X'].shape}, "
          f"{bundle['X'].shape[0]} patches over {len(cases_sorted)} cases")

    # --- Build split manifest for all 20 seeds ---
    seed_splits = {}
    for seed in range(1, 21):
        lu_train, lu_test = load_lu_split_for_seed(args.split_dir, seed)

        # Only keep cases that exist in our data
        train_cases = sorted(n for n in lu_train if n in all_cases)
        test_cases = sorted(n for n in lu_test if n in all_cases)

        # Split test into val + holdout
        val_cases, holdout_cases = stratify_test_split(
            test_cases, all_cases, val_frac=args.val_frac, seed=VAL_HOLDOUT_SEED,
        )

        seed_splits[str(seed)] = {
            "train": train_cases,
            "val": val_cases,
            "holdout": holdout_cases,
            "n_train": len(train_cases),
            "n_val": len(val_cases),
            "n_holdout": len(holdout_cases),
        }

    # Print summary
    for seed_num in range(1, 21):
        s = seed_splits[str(seed_num)]
        total = s["n_train"] + s["n_val"] + s["n_holdout"]
        print(f"  seed {seed_num:2d}: train={s['n_train']} val={s['n_val']} "
              f"hold={s['n_holdout']} total={total}")

    manifest = {
        "fmt_shape": args.fmt_shape,
        "val_holdout_seed": VAL_HOLDOUT_SEED,
        "val_frac": args.val_frac,
        "stratify_by": ["case_type", "density_bin", "wind_dir_bin"],
        "split_source": "Lu all_cases_20exp",
        "seeds": seed_splits,
    }
    manifest["content_hash"] = sha256_of(manifest)
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"Wrote split_manifest.json (hash={manifest['content_hash'][:12]})")
    print("Done!")


if __name__ == "__main__":
    main()
