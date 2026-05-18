#!/usr/bin/env python3
"""Validate materialized Auto V5 train_config.json files."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic import ValidationError

from shared.configs.schema import TrainConfig


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"ERROR: failed to parse JSON {path}: {exc}") from exc


def registry_names(repo_root: Path) -> set[str]:
    init_path = repo_root / "shared" / "models" / "__init__.py"
    tree = ast.parse(init_path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "register"
            and isinstance(func.value, ast.Name)
            and func.value.id == "REGISTRY"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
    return names


def included_architectures(repo_root: Path) -> set[str]:
    data = load_json(repo_root / "grids" / "v5_architectures.json")
    return {row["arch_name"] for row in data if row.get("include_in_v5", True)}


def train_configs(campaign: Path, run_id: str | None = None) -> list[Path]:
    if run_id:
        p = campaign / "runs" / run_id / "train_config.json"
        return [p] if p.is_file() else []
    if (campaign / "train_config.json").is_file():
        return [campaign / "train_config.json"]
    runs_dir = campaign / "runs"
    if not runs_dir.is_dir():
        return []
    return sorted(runs_dir.glob("*/train_config.json"))


def training_extras(cfg: dict[str, Any]) -> dict[str, Any]:
    arch_kwargs = cfg.get("arch_kwargs") or {}
    training = arch_kwargs.get("training") or {}
    if not isinstance(training, dict):
        return {}
    return training


def has_codegen_script_path(cfg: dict[str, Any]) -> bool:
    """Return true when a config intentionally uses a generated model file.

    Codegen smoke configs are not required to appear in the fixed V5
    architecture grid or built-in registry. They are guarded by script_path and
    later by static/dynamic model-file checks.
    """
    script_path = cfg.get("script_path")
    if not isinstance(script_path, str) or not script_path.endswith(".py"):
        return False
    normalized = script_path.replace("\\", "/")
    return "/generated_models/" in normalized or normalized.startswith("generated_models/")


def is_retry_config(cfg: dict[str, Any], cfg_path: Path) -> bool:
    """Return true for controller-created retry attempts.

    Benchmark200 source configs are intentionally constrained to the official
    HP grid, including batch_size=16. Resource retries may lower batch_size to
    8 after real OOM evidence while preserving epochs=200 and all other HPs.
    """
    experiment_id = cfg.get("experiment_id")
    if isinstance(experiment_id, str) and "_retry" in experiment_id:
        return True
    return "_retry" in cfg_path.parent.name


def validate_one(cfg_path: Path, cfg: dict[str, Any], *, stage: str, repo_root: Path) -> list[str]:
    errors: list[str] = []
    archs = included_architectures(repo_root)
    registry = registry_names(repo_root)
    extras = training_extras(cfg)
    codegen_script = has_codegen_script_path(cfg)

    def err(msg: str) -> None:
        errors.append(f"{cfg_path}: {msg}")

    try:
        TrainConfig.model_validate(cfg)
    except ValidationError as exc:
        extra_fields = sorted(
            str(item["loc"][0])
            for item in exc.errors()
            if item.get("type") == "extra_forbidden" and item.get("loc")
        )
        if extra_fields:
            err(f"extra fields not allowed by TrainConfig: {', '.join(extra_fields)}")
        else:
            for item in exc.errors():
                loc = ".".join(str(x) for x in item.get("loc", ())) or "<root>"
                err(f"TrainConfig schema validation failed at {loc}: {item.get('msg')}")

    arch = cfg.get("arch_name")
    if not codegen_script:
        if arch not in archs:
            err(f"arch_name {arch!r} not in V5 included architecture list")
        if arch not in registry:
            err(f"arch_name {arch!r} not registered in shared.models.REGISTRY")
    else:
        if not isinstance(arch, str) or not arch:
            err("codegen script_path configs must still provide a non-empty arch_name")

    expected_epochs = 20 if stage == "smoke20" else 200
    if cfg.get("epochs") != expected_epochs:
        err(f"epochs must be {expected_epochs} for {stage}, got {cfg.get('epochs')!r}")
    if int(cfg.get("batch_size", 999999)) > 16:
        err(f"batch_size must be <=16, got {cfg.get('batch_size')!r}")
    if cfg.get("compute_r2") is not True:
        err("compute_r2 must be true")
    if extras.get("data_augment", False) is not False:
        err("data_augment must be false")
    if cfg.get("input_features") != "height":
        err(f"input_features must be height, got {cfg.get('input_features')!r}")
    if cfg.get("seed") != 1:
        err(f"seed must be 1, got {cfg.get('seed')!r}")

    if stage == "benchmark200":
        hp = load_json(repo_root / "grids" / "v5_hp_candidates.json")
        allowed = hp.get("allowed", {})
        checks = {
            "loss_name": cfg.get("loss_name"),
            "lr": cfg.get("lr"),
            "batch_size": cfg.get("batch_size"),
            "epochs": cfg.get("epochs"),
            "compute_r2": cfg.get("compute_r2"),
            "input_features": cfg.get("input_features"),
            "seed": cfg.get("seed"),
            "data_augment": extras.get("data_augment", False),
            "ema": extras.get("ema_decay"),
            "scheduler": extras.get("scheduler"),
        }
        benchmark_retry = is_retry_config(cfg, cfg_path)
        for key, value in checks.items():
            if key == "batch_size" and benchmark_retry and value == 8:
                continue
            if key in allowed and value not in allowed[key]:
                err(f"{key}={value!r} not in allowed {allowed[key]!r}")
    return errors


def validate_campaign(campaign: Path, stage: str, repo_root: Path, run_id: str | None = None) -> list[str]:
    paths = train_configs(campaign, run_id=run_id)
    if not paths:
        scope = f"run_id={run_id}" if run_id else "campaign"
        return [f"{campaign}: no train_config.json files found for {scope}"]
    errors: list[str] = []
    for p in paths:
        cfg = load_json(p)
        if not isinstance(cfg, dict):
            errors.append(f"{p}: train_config root must be object")
            continue
        errors.extend(validate_one(p, cfg, stage=stage, repo_root=repo_root))
    if not errors:
        print(f"Validated {len(paths)} train configs")
        print("OK")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=Path, required=True)
    ap.add_argument("--stage", choices=["smoke20", "benchmark200"], required=True)
    ap.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    ap.add_argument("--run-id", default=None, help="Validate only campaigns/<name>/runs/<run-id>/train_config.json")
    args = ap.parse_args()
    errors = validate_campaign(args.campaign, args.stage, args.repo_root, run_id=args.run_id)
    if errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
