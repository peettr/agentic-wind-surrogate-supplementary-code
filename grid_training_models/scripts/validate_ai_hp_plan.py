#!/usr/bin/env python3
"""Validate an AI-curated Grid hyperparameter plan.

This script checks that every included architecture satisfies the configured
selection-policy count range, all HP values come from the selected HP policy
file, and all run IDs are unique. The policy may include per-architecture
overrides and an optional total_config_max.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REQUIRED_CONFIG_FIELDS = {
    "variant", "loss_name", "lr", "ema", "scheduler", "batch_size",
    "epochs", "data_augment", "compute_r2", "input_features", "seed", "reason",
}
ALLOWED_CONFIG_FIELDS = REQUIRED_CONFIG_FIELDS | {"notes", "protocol_deviation"}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"ERROR: failed to parse JSON {path}: {exc}") from exc


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_payload(arch_name: str, cfg: dict[str, Any]) -> str:
    payload = {
        k: cfg[k]
        for k in [
            "variant", "loss_name", "lr", "ema", "scheduler", "batch_size",
            "epochs", "data_augment", "compute_r2", "input_features", "seed",
        ]
    }
    payload["arch_name"] = arch_name
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def make_run_id(arch_name: str, variant: str, cfg: dict[str, Any]) -> str:
    safe_arch = re.sub(r"[^A-Za-z0-9_]+", "_", arch_name).strip("_")
    safe_variant = re.sub(r"[^A-Za-z0-9_]+", "_", variant).strip("_")
    h = hashlib.blake2b(canonical_payload(arch_name, cfg).encode(), digest_size=5).hexdigest()
    return f"r_{safe_arch}_{safe_variant}_{h}"


def load_registry_names(repo_root: Path) -> set[str]:
    """Return registered model names without importing torch-heavy modules.

    Importing shared.models requires torch and all model dependencies. The run
    generator should remain usable in a lightweight environment, so we parse
    REGISTRY.register("name", ...) calls from shared/models/__init__.py instead.
    """
    import ast

    init_path = repo_root / "shared" / "models" / "__init__.py"
    try:
        tree = ast.parse(init_path.read_text())
    except Exception as exc:
        raise SystemExit(f"ERROR: cannot parse {init_path}: {exc}") from exc

    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "register"
            and isinstance(func.value, ast.Name)
            and func.value.id == "REGISTRY"
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            names.add(node.args[0].value)
    return names


def validate(architectures: list[dict[str, Any]], hp: dict[str, Any], plan: list[dict[str, Any]], repo_root: Path) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    allowed = hp.get("allowed", {})
    policy = hp.get("selection_policy", {})
    min_cfg = int(policy.get("configs_per_arch_min", 3))
    max_cfg = int(policy.get("configs_per_arch_max", 5))
    per_arch_overrides = policy.get("per_arch_overrides", {}) or {}
    total_config_max = policy.get("total_config_max")
    if total_config_max is not None:
        total_config_max = int(total_config_max)
    if not isinstance(per_arch_overrides, dict):
        errors.append("selection_policy.per_arch_overrides must be an object when provided")
        per_arch_overrides = {}

    arch_by_name = {a.get("arch_name"): a for a in architectures}
    included = {a["arch_name"] for a in architectures if a.get("include_in_v5", True)}
    registry = load_registry_names(repo_root)

    for arch in sorted(included):
        if arch not in registry:
            errors.append(f"architecture not in shared.models.REGISTRY: {arch}")

    if not isinstance(plan, list):
        errors.append("plan root must be a list")
        return errors

    plan_by_arch: dict[str, dict[str, Any]] = {}
    for entry in plan:
        arch = entry.get("arch_name")
        if arch in plan_by_arch:
            errors.append(f"duplicate architecture entry in plan: {arch}")
        plan_by_arch[arch] = entry

    missing = sorted(included - set(plan_by_arch))
    extra = sorted(set(plan_by_arch) - included)
    if missing:
        errors.append(f"missing included architectures: {missing}")
    if extra:
        errors.append(f"plan contains unknown/not-included architectures: {extra}")

    run_ids: set[str] = set()
    total_configs = 0
    for arch, entry in sorted(plan_by_arch.items()):
        configs = entry.get("configs")
        if not isinstance(configs, list):
            errors.append(f"{arch}: configs must be a list")
            continue
        arch_policy = per_arch_overrides.get(arch, {})
        if arch_policy and not isinstance(arch_policy, dict):
            errors.append(f"{arch}: per-architecture override must be an object")
            arch_policy = {}
        arch_min_cfg = int(arch_policy.get("configs_per_arch_min", min_cfg))
        arch_max_cfg = int(arch_policy.get("configs_per_arch_max", max_cfg))
        if arch in included and not (arch_min_cfg <= len(configs) <= arch_max_cfg):
            errors.append(f"{arch}: expected {arch_min_cfg}-{arch_max_cfg} configs, got {len(configs)}")
        variants: set[str] = set()
        hp_payloads: set[str] = set()
        for i, cfg in enumerate(configs):
            total_configs += 1
            prefix = f"{arch}[{i}]"
            if not isinstance(cfg, dict):
                errors.append(f"{prefix}: config must be object")
                continue
            missing_fields = REQUIRED_CONFIG_FIELDS - set(cfg)
            extra_fields = set(cfg) - ALLOWED_CONFIG_FIELDS
            if missing_fields:
                errors.append(f"{prefix}: missing fields {sorted(missing_fields)}")
            if extra_fields:
                errors.append(f"{prefix}: unapproved extra fields {sorted(extra_fields)}")
            variant = cfg.get("variant")
            if not isinstance(variant, str) or not variant.strip():
                errors.append(f"{prefix}: variant must be non-empty string")
            elif variant in variants:
                errors.append(f"{arch}: duplicate variant {variant}")
            else:
                variants.add(variant)
            hp_payload = json.dumps(
                {
                    k: cfg.get(k)
                    for k in [
                        "loss_name", "lr", "ema", "scheduler", "batch_size",
                        "epochs", "data_augment", "compute_r2", "input_features", "seed",
                    ]
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if hp_payload in hp_payloads:
                errors.append(f"{prefix}: duplicate HP payload within architecture {arch}")
            hp_payloads.add(hp_payload)
            reason = cfg.get("reason")
            if not isinstance(reason, str) or len(reason.strip()) < 20:
                errors.append(f"{prefix}: reason must be a meaningful non-empty string")
            for field, vals in allowed.items():
                if field not in cfg:
                    continue
                if cfg[field] not in vals:
                    errors.append(f"{prefix}: {field}={cfg[field]!r} not in allowed {vals!r}")
            if cfg.get("batch_size", 0) > hp.get("hard_rules", {}).get("batch_size_max", 16):
                errors.append(f"{prefix}: batch_size exceeds max")
            if cfg.get("epochs") != 200:
                errors.append(f"{prefix}: epochs must be 200 for main V5 benchmark")
            if cfg.get("data_augment") is not False:
                errors.append(f"{prefix}: data_augment must be false")
            if cfg.get("compute_r2") is not True:
                errors.append(f"{prefix}: compute_r2 must be true")
            if cfg.get("input_features") != "height":
                errors.append(f"{prefix}: input_features must be height")
            if REQUIRED_CONFIG_FIELDS <= set(cfg):
                rid = make_run_id(arch, str(variant), cfg)
                if rid in run_ids:
                    errors.append(f"duplicate run_id generated: {rid}")
                run_ids.add(rid)

    if total_config_max is not None and total_configs > total_config_max:
        errors.append(f"total configs exceed total_config_max={total_config_max}: {total_configs}")

    if errors:
        return errors
    print(f"Validated {len(included)} architectures")
    print(f"Validated {total_configs} configs")
    print("No illegal HP values")
    print("No duplicate run IDs")
    print("OK")
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architectures", type=Path, required=True)
    ap.add_argument("--hp", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    args = ap.parse_args()

    errors = validate(load_json(args.architectures), load_json(args.hp), load_json(args.plan), args.repo_root)
    if errors:
        print("VALIDATION FAILED", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



