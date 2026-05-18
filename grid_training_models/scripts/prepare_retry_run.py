#!/usr/bin/env python3
"""Create a safe Grid retry run from an existing train_config.json.

Retry bookkeeping belongs in sidecars/manifest, never in train_config.json.
This script sanitizes known metadata fields, updates identity/path fields, and
validates the result against the strict TrainConfig schema before writing it.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.configs.schema import TrainConfig

BOOKKEEPING_FIELDS = {
    "run_id",
    "retry_of",
    "retry_reason",
    "retry_note",
    "retry_status",
    "retry_tier",
    "retry_cluster",
    "cluster_id",
    "idle_reason",
    "high_vram",
}


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: train_config root must be a JSON object")
    return data


def new_results_dir(old_results_dir: str, old_run_id: str, new_run_id: str) -> str:
    p = Path(old_results_dir)
    if p.name == old_run_id:
        return str(p.with_name(new_run_id))
    return str(p / new_run_id)


def prepare_retry_config(source_cfg: dict[str, Any], *, new_run_id: str, batch_size: int | None) -> dict[str, Any]:
    old_run_id = str(source_cfg.get("experiment_id", ""))
    cfg = {k: v for k, v in source_cfg.items() if k not in BOOKKEEPING_FIELDS}
    cfg["experiment_id"] = new_run_id
    if "results_dir" in cfg:
        cfg["results_dir"] = new_results_dir(str(cfg["results_dir"]), old_run_id, new_run_id)
    if batch_size is not None:
        if batch_size < 8:
            raise SystemExit("batch_size below Grid smoke lower bound (8) is not allowed")
        cfg["batch_size"] = batch_size
    TrainConfig.model_validate(cfg)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-config", type=Path, required=True)
    p.add_argument("--new-run-id", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--tier", required=True)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--base-run-id", default=None)
    p.add_argument("--attempt-index", type=int, default=None)
    p.add_argument("--classification", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_cfg = load_config(args.source_config)
    retry_cfg = prepare_retry_config(source_cfg, new_run_id=args.new_run_id, batch_size=args.batch_size)

    if args.output_dir is None:
        # source-config is normally <campaign>/runs/<old_run_id>/train_config.json
        output_dir = args.source_config.parents[1] / args.new_run_id
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(retry_cfg, indent=2) + "\n")

    note = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_config": str(args.source_config),
        "source_experiment_id": source_cfg.get("experiment_id"),
        "base_run_id": args.base_run_id,
        "attempt_index": args.attempt_index,
        "new_run_id": args.new_run_id,
        "classification": args.classification,
        "reason": args.reason,
        "tier": args.tier,
        "batch_size": retry_cfg.get("batch_size"),
        "metadata_rule": "retry bookkeeping is stored outside train_config.json",
    }
    note = {k: v for k, v in note.items() if v is not None}
    (output_dir / "RETRY_NOTE.txt").write_text("\n".join(f"{k}: {v}" for k, v in note.items()) + "\n")
    print(f"Wrote retry run {args.new_run_id} at {output_dir}")
    print("TrainConfig validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



