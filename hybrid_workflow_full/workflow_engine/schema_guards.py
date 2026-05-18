"""Shared schema guards for Hybrid planner/codegen/submit boundaries.

These guards mirror locked train.py/losses.py contracts without modifying them.
They intentionally reject aliases such as masked_l1_grad instead of normalizing.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALID_INPUT_FEATURE_CHANNELS: dict[str, int] = {
    "height": 1,
    "height_sdf": 2,
    "height_sdf_normal": 3,
}


def input_channels_for_features(input_features: str | None) -> int | None:
    return VALID_INPUT_FEATURE_CHANNELS.get(str(input_features or "height"))


def legal_loss_names() -> set[str]:
    """Return canonical loss registry names from shared/losses.py."""
    try:
        from shared.losses import LIBRARY as LOSS_LIBRARY  # read-only contract import
        return set(LOSS_LIBRARY.list_all())
    except ModuleNotFoundError as exc:
        # Local planner/schema runs may use a lightweight Python environment
        # without torch.  shared.losses imports torch before registering the
        # canonical loss names, but the names themselves are static source-level
        # contract data.  Fall back to AST extraction so schema validation does
        # not falsely reject every proposal solely because torch is unavailable.
        if exc.name != "torch":
            raise
        return legal_loss_names_from_source()


def legal_loss_names_from_source() -> set[str]:
    """Extract LIBRARY.register("name", ...) calls without importing torch."""
    path = PROJECT_ROOT / "shared" / "losses.py"
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (
            isinstance(fn, ast.Attribute)
            and fn.attr == "register"
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "LIBRARY"
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            names.add(node.args[0].value)
    if not names:
        raise RuntimeError(f"Could not extract loss registry names from {path}")
    return names


def _intish(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _floatish(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_experiment_schema(cfg: dict, *, stage: str = "submit") -> tuple[bool, list[dict]]:
    """Fail-closed validation for planner and submit configs.

    The goal is contract safety, not broad scientific judgment.  We only reject
    fields that are known to break locked train.py/losses.py or resource basics.
    """
    issues: list[dict] = []
    arch = cfg.get("arch_name")
    if not isinstance(arch, str) or not arch.strip():
        issues.append({"code": "MISSING_ARCH_NAME", "field": "arch_name", "value": arch})

    loss = cfg.get("loss_name", "masked_l1")
    try:
        losses = legal_loss_names()
    except Exception as exc:  # fail closed if the canonical registry cannot be read
        losses = set()
        issues.append({"code": "LOSS_REGISTRY_UNAVAILABLE", "field": "loss_name", "message": str(exc)})
    if loss not in losses:
        issues.append({"code": "LOSS_REGISTRY_KEYERROR", "field": "loss_name", "value": loss, "legal": sorted(losses)})

    features = cfg.get("input_features", "height")
    if input_channels_for_features(features) is None:
        issues.append({"code": "INVALID_INPUT_FEATURES", "field": "input_features", "value": features, "legal": sorted(VALID_INPUT_FEATURE_CHANNELS)})

    n_c = _intish(cfg.get("n_c", (cfg.get("arch_kwargs") or {}).get("n_c", 16)))
    depth = _intish(cfg.get("depth", (cfg.get("arch_kwargs") or {}).get("depth", 7)))
    batch = _intish(cfg.get("batch_size", 16))
    lr = _floatish(cfg.get("lr", 1e-3))
    if n_c is None or n_c < 1 or n_c > 512:
        issues.append({"code": "RESOURCE_BASIC_INVALID", "field": "n_c", "value": cfg.get("n_c")})
    if depth is None or depth < 1 or depth > 12:
        issues.append({"code": "ARCH_CONSTRAINT_FAIL", "field": "depth", "value": cfg.get("depth"), "message": "depth must be 1..12"})
    if batch is None or batch < 1 or batch > 128:
        issues.append({"code": "RESOURCE_BASIC_INVALID", "field": "batch_size", "value": cfg.get("batch_size")})
    if lr is None or lr <= 0 or lr > 1:
        issues.append({"code": "RESOURCE_BASIC_INVALID", "field": "lr", "value": cfg.get("lr")})

    # Evidence-backed registered arch constraint from R1 smoke stack/source:
    # quadmamba shared model exposes SUPPORTED_DEPTHS={5,6,7}; depth=4 raises ValueError.
    if arch == "quadmamba" and depth is not None and depth not in {5, 6, 7}:
        issues.append({
            "code": "ARCH_CONSTRAINT_FAIL",
            "field": "depth",
            "value": depth,
            "arch_name": arch,
            "legal": [5, 6, 7],
            "evidence": "shared/models/quadmamba.py SUPPORTED_DEPTHS and R1 smoke stack",
        })

    return len(issues) == 0, issues


def validate_registered_arch_config(arch_name: str, cfg: dict) -> tuple[bool, str]:
    """Conservative registered/shared architecture config validation.

    Only enforces known-safe checks: schema basics, quadmamba supported depths,
    and constructor compatibility for explicit arch_kwargs when the source file
    is importable. It does not instantiate arbitrary shared models here.
    """
    ok, issues = validate_experiment_schema(cfg, stage="codegen")
    if not ok:
        return False, "; ".join(f"{i.get('code')}:{i.get('field')}={i.get('value')}" for i in issues)
    model_path = PROJECT_ROOT / "shared" / "models" / f"{arch_name}.py"
    if not model_path.exists():
        return True, "ok"
    try:
        tree = ast.parse(model_path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return False, f"shared model syntax error: {exc}"
    class_names = {arch_name, ''.join(w.capitalize() for w in arch_name.split('_'))}
    class_node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in class_names), None)
    if class_node is None:
        return True, "ok"
    init = next((n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if init is None:
        return True, "ok"
    params = {a.arg for a in init.args.args if a.arg != "self"}
    has_kwargs = init.args.kwarg is not None
    base = {"in_channels", "out_channels", "n_c", "depth"}
    extra = {str(k) for k in (cfg.get("arch_kwargs") or {}) if str(k) not in base}
    missing = sorted(extra - params)
    if missing and not has_kwargs:
        return False, f"constructor does not accept arch_kwargs {missing}"
    return True, "ok"



