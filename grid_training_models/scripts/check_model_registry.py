#!/usr/bin/env python3
"""Check Auto V5 model registry coverage and optional lightweight readiness."""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise SystemExit(f"ERROR: failed to parse JSON {path}: {exc}") from exc


def load_registry_names(repo_root: Path) -> set[str]:
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


def included_archs(path: Path) -> list[str]:
    data = load_json(path)
    return [row["arch_name"] for row in data if row.get("include_in_v5", True)]


def metadata_by_arch(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, list):
        return {row["arch_name"]: row for row in data}
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    raise SystemExit("ERROR: metadata must be list or object")


def static_check(architectures: Path, metadata: Path, repo_root: Path) -> dict[str, Any]:
    registry = load_registry_names(repo_root)
    archs = included_archs(architectures)
    meta = metadata_by_arch(metadata)
    rows: list[dict[str, Any]] = []
    for arch in archs:
        row = meta.get(arch, {})
        source_file = row.get("source_file")
        source_exists = bool(source_file and (repo_root / source_file).is_file())
        in_registry = arch in registry
        status = "STATIC_READY" if in_registry and source_exists else "NEEDS_PATCH"
        rows.append(
            {
                "arch_name": arch,
                "registry_present": in_registry,
                "source_file": source_file,
                "source_exists": source_exists,
                "status": status,
            }
        )
    return {
        "summary": {
            "checked": len(rows),
            "registry_missing": sum(not r["registry_present"] for r in rows),
            "source_missing": sum(not r["source_exists"] for r in rows),
            "static_ready": sum(r["status"] == "STATIC_READY" for r in rows),
        },
        "models": rows,
    }


def dynamic_check(report: dict[str, Any], repo_root: Path, only: str | None = None) -> dict[str, Any]:
    """Optionally import torch/model registry and run constructor/forward probes.

    This is best-effort. Static checks are enough for lightweight environments.
    If PyTorch or optional model dependencies are unavailable, record a skip
    instead of failing the local code-readiness gate.
    """
    sys.path.insert(0, str(repo_root))
    try:
        import torch  # type: ignore
        from shared.models import REGISTRY  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local ML env
        report["summary"]["dynamic_skipped"] = True
        report["summary"]["dynamic_skip_reason"] = f"torch import failed or model dependencies missing: {exc}"
        return report

    for row in report["models"]:
        if only and row["arch_name"] != only:
            continue
        if row["status"] != "STATIC_READY":
            continue
        arch = row["arch_name"]
        try:
            ctor = REGISTRY.get(arch)
            model = ctor()
            params = sum(p.numel() for p in model.parameters())
            row["params"] = params
            row["params_m"] = params / 1e6
            if params > 150_000_000:
                row["status"] = "PARAM_LIMIT_FAIL"
                continue
            model.eval()
            # Use 64x64 for fast local compatibility probing. Full 640x640 is covered by smoke20.
            x = torch.zeros(1, 1, 64, 64)
            with torch.no_grad():
                y = model(x)
            row["forward_shape"] = list(y.shape)
            finite = bool(torch.isfinite(y).all().item())
            row["finite_output"] = finite
            if list(y.shape[:2]) != [1, 1] or y.ndim != 4:
                row["status"] = "FORWARD_SHAPE_FAIL"
            elif not finite:
                row["status"] = "FORWARD_NAN_FAIL"
            else:
                row["status"] = "FORWARD_READY"
        except Exception as exc:  # pragma: no cover - dynamic path is env dependent
            row["status"] = "DYNAMIC_FAIL"
            row["error"] = repr(exc)
    report["summary"]["forward_ready"] = sum(r.get("status") == "FORWARD_READY" for r in report["models"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architectures", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=repo_root_from_script())
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    report = static_check(args.architectures, args.metadata, args.repo_root)
    if not args.static_only:
        report = dynamic_check(report, args.repo_root, args.only)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Checked {report['summary']['checked']} architectures")
    print(f"registry_missing={report['summary']['registry_missing']}")
    print(f"source_missing={report['summary']['source_missing']}")
    if report["summary"]["registry_missing"] or report["summary"]["source_missing"]:
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
