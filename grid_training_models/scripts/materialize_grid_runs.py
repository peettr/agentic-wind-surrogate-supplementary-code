#!/usr/bin/env python3
"""Materialize an AI-curated Grid HP plan into executable run directories."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

# Import run_id helper from sibling validator to guarantee identical IDs.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from validate_ai_hp_plan import load_json, validate, make_run_id, repo_root_from_script  # noqa: E402


def train_config_for_run(
    *,
    run_id: str,
    arch_name: str,
    variant: str,
    cfg: dict[str, Any],
    campaign_dir: Path,
    remote_root: str,
    strategy: str,
) -> dict[str, Any]:
    remote_root = remote_root.rstrip("/")
    rel_results = f"campaigns/{campaign_dir.name}/runs/{run_id}"
    training_extras = {
        "scheduler": cfg.get("scheduler"),
        "ema_decay": cfg.get("ema"),
        "data_augment": cfg.get("data_augment", False),
    }
    # Keep only non-None training extras except data_augment, which is a hard-rule record.
    training_extras = {k: v for k, v in training_extras.items() if v is not None or k == "data_augment"}
    arch_kwargs: dict[str, Any] = {}
    if training_extras:
        arch_kwargs["training"] = training_extras

    return {
        "experiment_id": run_id,
        "strategy": strategy,
        "seed": cfg["seed"],
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "checkpoint_interval": 50,
        "arch_name": arch_name,
        "arch_kwargs": arch_kwargs,
        "loss_name": cfg["loss_name"],
        "loss_kwargs": {},
        "data_dir": f"{remote_root}/shared/data",
        "results_dir": f"{remote_root}/{rel_results}",
        "split_manifest_path": f"{remote_root}/shared/data/split_manifest.json",
        "heartbeat_interval_epochs": 10,
        "compute_r2": cfg["compute_r2"],
        "phase": "search",
        "eval_splits": ["val"],
        "script_path": f"{remote_root}/shared/train.py",
        "early_stop_wall_min": 100,
        "max_wall_min": 200,
        "baseline_r2_curve_path": f"{remote_root}/shared/data/baseline_r2_curve.json",
        "input_features": cfg["input_features"],
    }


def materialize(
    architectures_path: Path,
    hp_path: Path,
    plan_path: Path,
    campaign_dir: Path,
    remote_root: str,
    repo_root: Path,
    strategy: str,
    overwrite: bool,
) -> None:
    archs = load_json(architectures_path)
    hp = load_json(hp_path)
    plan = load_json(plan_path)
    errors = validate(archs, hp, plan, repo_root)
    if errors:
        raise SystemExit(1)

    if campaign_dir.exists() and overwrite:
        shutil.rmtree(campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = campaign_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    for entry in plan:
        arch_name = entry["arch_name"]
        for cfg in entry["configs"]:
            variant = cfg["variant"]
            run_id = make_run_id(arch_name, variant, cfg)
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            train_cfg = train_config_for_run(
                run_id=run_id,
                arch_name=arch_name,
                variant=variant,
                cfg=cfg,
                campaign_dir=campaign_dir,
                remote_root=remote_root,
                strategy=strategy,
            )
            (run_dir / "train_config.json").write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")
            (run_dir / "reason.txt").write_text(cfg["reason"].strip() + "\n", encoding="utf-8")
            rows.append({
                "run_id": run_id,
                "arch_name": arch_name,
                "variant": variant,
                "config_path": str(Path("runs") / run_id / "train_config.json"),
                "reason_path": str(Path("runs") / run_id / "reason.txt"),
                "reason": cfg["reason"],
                "loss_name": cfg["loss_name"],
                "lr": cfg["lr"],
                "ema": cfg["ema"],
                "scheduler": cfg["scheduler"],
            })
    (campaign_dir / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Materialized {len(rows)} runs into {campaign_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architectures", type=Path, required=True)
    ap.add_argument("--hp", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--campaign-dir", type=Path, required=True)
    ap.add_argument("--remote-root", default="<GRID_HPC_SOURCE_ROOT>")
    ap.add_argument("--strategy", default="grid_curated")
    ap.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    materialize(args.architectures, args.hp, args.plan, args.campaign_dir, args.remote_root, args.repo_root, args.strategy, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



