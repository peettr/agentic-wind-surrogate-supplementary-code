#!/usr/bin/env python3
"""Campaign-level Auto V5 orchestrator plan builder.

This layer owns deterministic campaign planning across smoke and benchmark stages.
It is safe by default: plan-only, no CRC SSH, no Condor submit, no repair agent.
Live submission remains an explicit opt-in handled by lower-level launch scripts.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

HIGH_TIER_TERMS = ("fno", "fourier", "spectral", "afno", "ufno", "ffno", "kan", "multiscale", "naf", "mamba")
DEFAULT_REMOTE_ROOT = "<GRID_HPC_SOURCE_ROOT>"
DEFAULT_HARD_EXCLUDE_ARCHES = {
    "umamba",  # frontend dynamic shape/runtime failure with default registry wrapper
    "unet_sdf_7level",  # frontend dynamic input-channel contract failure with default registry wrapper
}
DEFAULT_SOFT_EXCLUDE_ARCHES = {
    "unet_v2_baseline",
    "attention_gate_unet",
    "cno_v2",
    "cbam_unet",
    "sac_unet",
    "dcn_unet",
    "fno2d",
    "fourier_unet",
    "kan_unet",
    "multiscale_conv",
}
RELAXED_PARAM_CAP_ARCHES = {
    "dilated_unet",  # historical frontend preflight params 497,756,193 > 150M with curated HP source
    "hrformer",  # historical frontend preflight params 215,350,593 > 150M with curated HP source
    "mamba2d",  # historical frontend preflight params 423,114,785 > 150M with default registry wrapper
    "quadmamba",  # historical frontend preflight params 338,380,833 > 150M with default registry wrapper
    "residual_spectral",  # historical frontend preflight params 171,032,545 > 150M with default registry wrapper
    "ufno",  # historical frontend preflight params 242,225,153 > 150M with default registry wrapper
    "unet_afno",  # historical frontend preflight params 188,782,610 > 150M with default registry wrapper
}
ARCH_SOURCE_FILE_ALIASES = {
    "cno_v2": "cno",
    "perceiver_io": "perceiver",
    "fno2d": "fno2d",
    "fno_v3": "fno_v3",
    "unet_v3_5level": "unet_v3",
    "unet_v3_6level": "unet_v3",
    "unet_v3_7level": "unet_v3",
}
# Backward-compatible name: only true hard failures remain absolute excludes.
DEFAULT_EXCLUDE_ARCHES = DEFAULT_HARD_EXCLUDE_ARCHES


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "run"


def is_submit_enabled(*, materialize: bool, submit: bool, dry_run: bool) -> bool:
    return bool(materialize and submit and not dry_run)


def _hp_reason(row: dict[str, Any]) -> str:
    parts = []
    if row.get("loss_name") is not None:
        parts.append(f"loss={row['loss_name']}")
    if row.get("lr") is not None:
        parts.append(f"lr={row['lr']}")
    if row.get("batch_size") is not None:
        parts.append(f"batch={row['batch_size']}")
    if row.get("reason"):
        parts.append(str(row["reason"]))
    return "; ".join(parts) if parts else "deterministically selected Auto V5 candidate"


def _tier_for_arch(arch: str, row: dict[str, Any]) -> str:
    if row.get("submit_tier"):
        return str(row["submit_tier"])
    lower = arch.lower()
    if any(term in lower for term in HIGH_TIER_TERMS):
        return "h100_a100_l40s_12gb"
    return "a40_rtx6k_16gb"


def build_smoke_control_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    campaign: str,
    run_prefix: str,
    count: int = 10,
    exclude_arches: set[str] | None = None,
    soft_exclude_arches: set[str] | None = None,
    exclude_sources: set[tuple[str, str]] | None = None,
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> dict[str, Any]:
    exclude_arches = set(exclude_arches or set())
    soft_exclude_arches = set(soft_exclude_arches or set())
    exclude_sources = set(exclude_sources or set())
    sorted_candidates = sorted(candidates, key=lambda r: (str(r.get("arch_name", "")), str(r.get("source_run_id", ""))))
    selected: list[dict[str, Any]] = []
    seen_arches: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()

    def add_unique(*, allow_soft: bool) -> None:
        for row in sorted_candidates:
            arch = str(row.get("arch_name") or row.get("arch") or "").strip()
            source_run_id = str(row.get("source_run_id", ""))
            source_campaign = str(row.get("source_campaign", ""))
            source_key = (source_campaign, source_run_id)
            if source_key in exclude_sources:
                continue
            if not arch or arch in exclude_arches or arch in seen_arches:
                continue
            if not allow_soft and arch in soft_exclude_arches:
                continue
            selected.append(row)
            seen_arches.add(arch)
            seen_sources.add((arch, source_run_id))
            if len(selected) == count:
                return

    def add_additional_hp(*, allow_soft: bool) -> None:
        for row in sorted_candidates:
            arch = str(row.get("arch_name") or row.get("arch") or "").strip()
            source_run_id = str(row.get("source_run_id", ""))
            source_campaign = str(row.get("source_campaign", ""))
            key = (arch, source_run_id)
            source_key = (source_campaign, source_run_id)
            if source_key in exclude_sources:
                continue
            if not arch or arch in exclude_arches or key in seen_sources:
                continue
            if not allow_soft and arch in soft_exclude_arches:
                continue
            selected.append(row)
            seen_sources.add(key)
            if len(selected) == count:
                return

    add_unique(allow_soft=False)
    if len(selected) < count:
        add_unique(allow_soft=True)
    if len(selected) < count:
        add_additional_hp(allow_soft=False)
    if len(selected) < count:
        add_additional_hp(allow_soft=True)
    if len(selected) != count:
        raise SystemExit(f"not enough candidates after exclusions: need {count}, got {len(selected)}")

    runs = []
    for idx, row in enumerate(selected):
        arch = str(row.get("arch_name") or row.get("arch"))
        slug = slugify(arch)
        run_id = f"{run_prefix}_{idx:02d}_{slug}_smoke20"
        out = {
            "source_campaign": str(row["source_campaign"]),
            "source_run_id": str(row["source_run_id"]),
            "run_id": run_id,
            "arch_name": arch,
            "model_file": str(row["model_file"]),
            "module_name": str(row["module_name"]),
            "submit_tier": _tier_for_arch(arch, row),
            "reason": _hp_reason(row),
        }
        batch = row.get("batch_size")
        if any(term in arch.lower() for term in HIGH_TIER_TERMS):
            out["batch_size"] = max(8, min(int(batch or 8), 8))
        elif batch is not None and int(batch) != 16:
            out["batch_size"] = int(batch)
        if arch in RELAXED_PARAM_CAP_ARCHES or arch in soft_exclude_arches:
            out["allow_param_cap_relaxation"] = True
            out["reason"] += "; param-cap relaxation enabled by round policy"
        runs.append(out)

    return {
        "campaign": campaign,
        "stage": "smoke20",
        "remote_root": remote_root,
        "description": "Campaign-level Auto V5 orchestrator smoke control generated deterministically from architecture and HP candidates. Safe plan-only by default.",
        "config_overrides": {"epochs": 20, "strategy": "v5_controller_auto10_smoke20_codegen"},
        "runs": runs,
    }


def merge_retry_passes_for_promotion(
    smoke_control: dict[str, Any],
    controller_state: dict[str, Any],
    retry_states: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a promotion view where original RETRY rows use passing retry attempts.

    The monitor keeps original smoke and retry1 state in separate files. Benchmark
    promotion, however, expects one state object keyed by the original smoke run id.
    This helper preserves original PASS rows and rewrites RETRY rows to PASS when
    their planned retry run reached PASS:RECORD_RESULT.
    """
    merged = {"runs": dict(controller_state.get("runs", {}))}
    retry_runs: dict[str, Any] = {}
    for state in retry_states:
        retry_runs.update(state.get("runs", {}))

    for row in smoke_control.get("runs", []):
        run_id = row["run_id"]
        state_row = merged["runs"].get(run_id, {})
        if state_row.get("state_key") == "PASS:RECORD_RESULT":
            continue
        retry_id = str((state_row.get("plan") or {}).get("new_run_id") or f"{run_id}_retry1")
        retry_row = retry_runs.get(retry_id)
        if retry_row and retry_row.get("state_key") == "PASS:RECORD_RESULT":
            promoted = dict(state_row)
            for key in ("classification", "last_evidence_summary", "plan", "cluster_id", "current_run_id"):
                if key in retry_row:
                    promoted[key] = retry_row[key]
            promoted["state_key"] = "PASS:RECORD_RESULT"
            promoted["current_run_id"] = str(retry_row.get("current_run_id") or retry_id)
            promoted["promotion_source"] = "retry_state"
            promoted["promotion_retry_run_id"] = retry_id
            merged["runs"][run_id] = promoted
    return merged


def build_benchmark_control_from_smoke_passes(
    smoke_control: dict[str, Any],
    controller_state: dict[str, Any],
    *,
    campaign: str,
    run_prefix: str,
) -> dict[str, Any]:
    states = controller_state.get("runs", {})
    if not isinstance(states, dict):
        raise SystemExit("controller state missing runs object")
    not_passed = [row["run_id"] for row in smoke_control["runs"] if states.get(row["run_id"], {}).get("state_key") != "PASS:RECORD_RESULT"]
    if not_passed:
        raise SystemExit("not all smoke runs passed: " + ", ".join(not_passed))

    runs = []
    for idx, row in enumerate(smoke_control["runs"]):
        arch = str(row.get("arch_name") or row["run_id"].replace("_smoke20", ""))
        slug = slugify(arch)
        state_row = states.get(row["run_id"], {})
        # Smoke20 is a code/runtime validation gate only. Do not use smoke-stage
        # validation metrics or ad-hoc promotion_allowed flags as performance
        # filters; benchmark200 is the first stage where model quality is judged.
        source_run_id = str(state_row.get("current_run_id") or row["run_id"])
        out = {
            "source_campaign": smoke_control["campaign"],
            "source_run_id": source_run_id,
            "run_id": f"{run_prefix}_{idx:02d}_{slug}_benchmark200",
            "arch_name": arch,
            "model_file": row["model_file"],
            "module_name": row["module_name"],
            "submit_tier": row["submit_tier"],
            "reason": "promoted after smoke code validation PASS:RECORD_RESULT; benchmark policy restores stage defaults where applicable",
        }
        runs.append(out)
    return {
        "campaign": campaign,
        "stage": "benchmark200",
        "remote_root": smoke_control.get("remote_root", DEFAULT_REMOTE_ROOT),
        "description": "Benchmark200 promotion control generated only after every smoke run passed controller monitoring.",
        "config_overrides": {"epochs": 200, "batch_size": 16, "strategy": "v5_benchmark200_codegen"},
        "runs": runs,
    }


def _arch_from_run_id(run_id: str) -> str:
    run_id = run_id.replace("_smoke20", "").replace("_benchmark200", "")
    parts = run_id.split("_")
    if len(parts) >= 3 and parts[1].startswith("auto") and parts[2].isdigit():
        return "_".join(parts[3:])
    return run_id


def collect_arches_from_control_files(paths: list[Path]) -> set[str]:
    arches: set[str] = set()
    for path in paths:
        if not path.exists():
            raise SystemExit(f"exclude control not found: {path}")
        data = json.loads(path.read_text())
        for row in data.get("runs", []):
            arch = str(row.get("arch_name") or "").strip()
            if not arch and row.get("run_id"):
                arch = _arch_from_run_id(str(row["run_id"]))
            if arch:
                arches.add(arch)
    return arches


def collect_sources_from_control_files(paths: list[Path]) -> set[tuple[str, str]]:
    sources: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            raise SystemExit(f"exclude control not found: {path}")
        data = json.loads(path.read_text())
        for row in data.get("runs", []):
            source_campaign = str(row.get("source_campaign") or "").strip()
            source_run_id = str(row.get("source_run_id") or "").strip()
            if source_campaign and source_run_id:
                sources.add((source_campaign, source_run_id))
    return sources


def _plan_from_control(control: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign": control["campaign"],
        "remote_root": control.get("remote_root", DEFAULT_REMOTE_ROOT),
        "stage": control.get("stage", "smoke20"),
        "summary": {"runs": len(control["runs"])},
        "runs": [
            {
                "run_id": row["run_id"],
                "source_campaign": row["source_campaign"],
                "source_run_id": row["source_run_id"],
                "model_file": row["model_file"],
                "module_name": row["module_name"],
                "submit_tier": row["submit_tier"],
                "batch_size": row.get("batch_size"),
                "command": "bash scripts/crc_codegen_smoke_one.sh",
                "command_env": {
                    "CAMPAIGN": control["campaign"],
                    "RUN_ID": row["run_id"],
                    "MODEL_FILE": row["model_file"],
                    "MODULE_NAME": row["module_name"],
                    "SUBMIT_TIER": row["submit_tier"],
                    "REMOTE": control.get("remote_root", DEFAULT_REMOTE_ROOT),
                    "STAGE": control.get("stage", "smoke20"),
                },
            }
            for row in control["runs"]
        ],
    }


EMBEDDED_BASE_SURROGATE = '''
from abc import ABC, abstractmethod


class BaseSurrogate(nn.Module, ABC):
    """Standalone BaseSurrogate copy for generated models."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for Auto V5 generated training source-of-truth models."""

    def check_shapes(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if x.shape[1:] != (1, 640, 640):
            raise ValueError(f"Input shape mismatch: expected (B, 1, 640, 640), got {tuple(x.shape)}")
        if y.shape[1:] != (1, 640, 640):
            raise ValueError(f"Output shape mismatch: expected (B, 1, 640, 640), got {tuple(y.shape)}")
'''


def _standalone_model_class_name(source_text: str, source_path: Path) -> str:
    tree = ast.parse(source_text, filename=str(source_path))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseSurrogate":
                return node.name
            if isinstance(base, ast.Attribute) and base.attr == "BaseSurrogate":
                return node.name
    raise SystemExit(f"{source_path}: no class deriving from BaseSurrogate found for standalone generation")


STANDALONE_RELATIVE_DEP_ALLOWLIST = {"afno_block"}


def _strip_module_main_block(source_text: str) -> str:
    return re.sub(r"\n\nif __name__ == ['\"]__main__['\"]:\n(?:    .*\n?)+\Z", "\n", source_text, flags=re.M)


def _render_standalone_relative_dependency(*, source_path: Path, module_name: str) -> str:
    """Return an embedded copy of a simple shared-model dependency module."""
    dep_path = source_path.parent / f"{module_name}.py"
    if not dep_path.exists():
        raise SystemExit(f"{source_path}: relative dependency not found for standalone generation: {dep_path}")
    dep_text = dep_path.read_text()
    forbidden_tokens = ("from shared.models", "import shared.models", "REGISTRY", "MODEL_REGISTRY")
    for token in forbidden_tokens:
        if token in dep_text:
            raise SystemExit(f"{dep_path}: cannot embed standalone dependency containing {token!r}")
    nested = re.findall(r"^from \.([A-Za-z0-9_]+) import (.+)$", dep_text, flags=re.M)
    nested_unsupported = [(mod, names) for mod, names in nested if mod != "base"]
    if nested_unsupported:
        preview = "; ".join(f"from .{mod} import {names}" for mod, names in nested_unsupported)
        raise SystemExit(f"{dep_path}: unsupported nested shared relative imports for standalone generation: {preview}")
    dep_text = re.sub(r"^from __future__ import annotations\r?\n", "", dep_text, flags=re.M)
    dep_text = re.sub(r"^from \.base import BaseSurrogate\r?\n", "", dep_text, flags=re.M)
    dep_text = _strip_module_main_block(dep_text).strip()
    return f"\n\n# Embedded local dependency copy: {module_name}.py\n{dep_text}\n"


def render_standalone_model_copy(*, arch_name: str, source_path: Path) -> str:
    """Copy a shared model implementation into a generated standalone model file.

    The generated file is the runtime training source of truth. It may be copied
    from shared/models, but it must not import shared.models, call REGISTRY, or use
    relative shared model functions at training time.
    """
    source_text = source_path.read_text()
    model_class = _standalone_model_class_name(source_text, source_path)
    forbidden_tokens = ("from shared.models", "import shared.models", "REGISTRY", "MODEL_REGISTRY")
    for token in forbidden_tokens:
        if token in source_text:
            raise SystemExit(f"{source_path}: cannot generate standalone model from source containing {token!r}")
    relative_imports = re.findall(r"^from \.([A-Za-z0-9_]+) import (.+)$", source_text, flags=re.M)
    unsupported = [
        (mod, names)
        for mod, names in relative_imports
        if mod != "base" and mod not in STANDALONE_RELATIVE_DEP_ALLOWLIST
    ]
    if unsupported:
        preview = "; ".join(f"from .{mod} import {names}" for mod, names in unsupported)
        raise SystemExit(f"{source_path}: unsupported shared relative imports for standalone generation: {preview}")
    standalone = re.sub(r"^from \.base import BaseSurrogate\r?\n", EMBEDDED_BASE_SURROGATE + "\n", source_text, flags=re.M)
    for mod, _names in relative_imports:
        if mod in STANDALONE_RELATIVE_DEP_ALLOWLIST:
            dep_text = _render_standalone_relative_dependency(source_path=source_path, module_name=mod)
            standalone = re.sub(rf"^from \.{re.escape(mod)} import .+\r?\n", dep_text + "\n", standalone, count=1, flags=re.M)
    if arch_name == "umamba":
        # The shared UMamba fallback SSM block projects each channel state to a
        # full dim-vector, then reshapes to (B, dim). The generated standalone
        # dynamic-check path needs the intended per-channel scalar projection.
        standalone = standalone.replace(
            "self.C_proj = nn.Linear(d_state, dim, bias=False)",
            "self.C_proj = nn.Linear(d_state, 1, bias=False)",
        )
        # The library default width produces >300M parameters when no arch_kwargs
        # are supplied by the curated source config. Keep generated initial runs
        # within the formal smoke/benchmark model-size guardrail by using a
        # narrower standalone default while preserving explicit user/config kwargs.
        standalone = standalone.replace(
            "def __init__(self, depth: int = 7, n_c: int = 32, d_state: int = 16,\n                 n_ssm_blocks: int = 4) -> None:",
            "def __init__(self, depth: int = 7, n_c: int = 16, d_state: int = 16,\n                 n_ssm_blocks: int = 2) -> None:",
        )
    header = (
        f'"""Generated standalone Auto V5 model for {arch_name}.\n\n'
        'This generated file is the training source of truth for this run.\n'
        'Runtime model construction is local to this file rather than registry delegation.\n'
        '"""\n'
    )
    standalone = re.sub(r'^""".*?"""\s*\n', header, standalone, count=1, flags=re.S)
    if standalone == source_text:
        standalone = header + standalone
    if "class Model(" not in standalone:
        standalone = re.sub(r"\n\nModel = [A-Za-z_][A-Za-z0-9_]*\s*$", "", standalone.rstrip(), count=1)
        model_body = (
            f"\n\n\nclass Model({model_class}):\n"
            "    \"\"\"Training entrypoint for generated Auto V5 runs.\"\"\"\n\n"
            "    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):\n"
            "        kwargs.pop('training', None)\n"
        )
        if arch_name == "unet_sdf_7level":
            model_body += "        kwargs.setdefault('in_channels', in_channels)\n"
        model_body += "        super().__init__(**kwargs)\n"
        if arch_name == "cnn_deeponet":
            model_body += (
                "\n"
                "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
                "        out = super().forward(x)\n"
                "        if out.shape[-2:] != x.shape[-2:]:\n"
                "            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)\n"
                "        return out\n"
            )
        standalone = standalone.rstrip() + model_body
    return standalone


def materialize_codegen_wrappers(local_root: Path, smoke_control: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for row in smoke_control["runs"]:
        arch = str(row["arch_name"])
        model_path = local_root / row["model_file"]
        source_arch = ARCH_SOURCE_FILE_ALIASES.get(arch, arch)
        source_path = local_root / "shared" / "models" / f"{source_arch}.py"
        if not source_path.exists():
            raise SystemExit(f"{arch}: shared source model not found for standalone generation: {source_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(render_standalone_model_copy(arch_name=arch, source_path=source_path))
        paths.append(model_path)
    return paths


def write_campaign_artifacts(
    *,
    report_dir: Path,
    smoke_control: dict[str, Any],
    materialize: bool = False,
    submit: bool = False,
    live_crc: bool = False,
    execute_repair: bool = False,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    control_path = report_dir / "control_smoke20.json"
    plan_path = report_dir / "launch_plan_smoke20.json"
    control_path.write_text(json.dumps(smoke_control, indent=2) + "\n")
    plan = _plan_from_control(smoke_control)
    plan["safety"] = {
        "materialize": bool(materialize),
        "submit": bool(submit),
        "live_crc": bool(live_crc),
        "execute_repair": bool(execute_repair),
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return {"smoke_control": control_path, "smoke_plan": plan_path}


def _candidate_from_source_config(path: Path, root: Path, *, model_dir: str) -> dict[str, Any] | None:
    cfg = json.loads(path.read_text())
    arch = cfg.get("arch_name")
    if not arch:
        return None
    run_id = path.parent.name
    campaign = path.parents[2].name
    slug = slugify(str(arch))
    model_file = f"{model_dir.rstrip('/')}/{slug}.py"
    return {
        "source_campaign": campaign,
        "source_run_id": run_id,
        "arch_name": str(arch),
        "model_file": model_file,
        "module_name": f"codegen_{slug}",
        "submit_tier": _tier_for_arch(str(arch), {}),
        "batch_size": cfg.get("batch_size", 16),
        "loss_name": cfg.get("loss_name"),
        "lr": cfg.get("lr"),
        "reason": "selected from existing curated HP train_config; wrapper generated by campaign orchestrator",
    }


def load_default_candidates(root: Path, *, model_dir: str, source_campaign: str = "v5_ai_curated_001") -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    source_runs = root / "campaigns" / source_campaign / "runs"
    for path in sorted(source_runs.glob("*/train_config.json")):
        row = _candidate_from_source_config(path, root, model_dir=model_dir)
        if row is not None:
            candidates.append(row)
    if not candidates:
        raise SystemExit(f"no default candidates found in {source_runs}")
    return candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-root", type=Path, default=_REPO_ROOT)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--campaign", default="v5_controller_auto10_001_smoke20")
    p.add_argument("--run-prefix", default="r_auto10")
    p.add_argument("--count", type=int, default=10)
    p.add_argument(
        "--source-campaign",
        default="v5_ai_curated_001",
        help="curated source campaign to sample candidates from, for example v5_ai_curated_002",
    )
    p.add_argument(
        "--include-hard-excluded-arches",
        action="store_true",
        help="include known hard-excluded arches when a bounded source pool must be exhausted; expect repair/failed classifications if they still fail",
    )
    p.add_argument("--dry-run", action="store_true", help="plan only; never submit")
    p.add_argument("--execute-codegen", action="store_true", help="write deterministic generated wrapper files for selected architectures")
    p.add_argument("--materialize", action="store_true", help="reserved explicit opt-in for lower-level materialization")
    p.add_argument("--submit-smoke", action="store_true", help="reserved explicit opt-in for smoke submit")
    p.add_argument("--live-crc", action="store_true", help="reserved explicit opt-in for live CRC monitoring")
    p.add_argument(
        "--model-dir",
        default="generated_models/v5_controller_auto10_001",
        help="directory for deterministic wrappers, for example generated_models/v5_controller_auto10_002",
    )
    p.add_argument(
        "--exclude-control",
        action="append",
        type=Path,
        default=[],
        help="previous control file whose arch_name/run_id entries should be excluded from this batch; may be repeated",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    candidates = load_default_candidates(args.local_root, model_dir=args.model_dir, source_campaign=args.source_campaign)
    hard_exclude_arches = set() if args.include_hard_excluded_arches else DEFAULT_HARD_EXCLUDE_ARCHES
    soft_exclude_arches = DEFAULT_SOFT_EXCLUDE_ARCHES | collect_arches_from_control_files(args.exclude_control)
    exclude_sources = collect_sources_from_control_files(args.exclude_control)
    control = build_smoke_control_from_candidates(
        candidates,
        campaign=args.campaign,
        run_prefix=args.run_prefix,
        count=args.count,
        exclude_arches=hard_exclude_arches,
        soft_exclude_arches=soft_exclude_arches,
        exclude_sources=exclude_sources,
    )
    outputs = write_campaign_artifacts(
        report_dir=args.output_dir,
        smoke_control=control,
        materialize=args.materialize,
        submit=is_submit_enabled(materialize=args.materialize, submit=args.submit_smoke, dry_run=args.dry_run),
        live_crc=args.live_crc,
    )
    generated_wrappers = materialize_codegen_wrappers(args.local_root, control) if args.execute_codegen else []
    payload = {
        "mode": "plan-only" if not is_submit_enabled(materialize=args.materialize, submit=args.submit_smoke, dry_run=args.dry_run) else "submit-enabled",
        "campaign": control["campaign"],
        "outputs": {k: str(v) for k, v in outputs.items()},
        "generated_wrappers": [str(p.relative_to(args.local_root)) for p in generated_wrappers],
        "safety": {
            "materialize": bool(args.materialize),
            "execute_codegen": bool(args.execute_codegen),
            "submit": is_submit_enabled(materialize=args.materialize, submit=args.submit_smoke, dry_run=args.dry_run),
            "live_crc": bool(args.live_crc),
            "execute_repair": False,
        },
        "hard_excluded_arch_count": len(hard_exclude_arches),
        "soft_excluded_arch_count": len(soft_exclude_arches),
        "runs": [row["run_id"] for row in control["runs"]],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
