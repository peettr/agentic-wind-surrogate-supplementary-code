#!/usr/bin/env python3
"""Campaign-level Auto V5 orchestrator plan builder.

This layer owns deterministic campaign planning across smoke and benchmark stages.
It is safe by default: plan-only, no CRC SSH, no Condor submit, no repair agent.
Live submission remains an explicit opt-in handled by lower-level launch scripts.
"""
from __future__ import annotations

import argparse
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
DEFAULT_EXCLUDE_ARCHES = {
    "unet_v2_baseline",
    "attention_gate_unet",
    "cno_v2",
    "cbam_unet",
    "sac_unet",
    "dcn_unet",
    "dilated_unet",  # frontend preflight params 497,756,193 > 150M with curated HP source
    "hrformer",  # frontend preflight params 215,350,593 > 150M with curated HP source
    "fno2d",
    "fourier_unet",
    "kan_unet",
    "multiscale_conv",
}


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
    remote_root: str = DEFAULT_REMOTE_ROOT,
) -> dict[str, Any]:
    exclude_arches = set(exclude_arches or set())
    selected: list[dict[str, Any]] = []
    seen_arches: set[str] = set()
    for row in sorted(candidates, key=lambda r: (str(r.get("arch_name", "")), str(r.get("source_run_id", "")))):
        arch = str(row.get("arch_name") or row.get("arch") or "").strip()
        if not arch or arch in exclude_arches or arch in seen_arches:
            continue
        selected.append(row)
        seen_arches.add(arch)
        if len(selected) == count:
            break
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
        runs.append(out)

    return {
        "campaign": campaign,
        "stage": "smoke20",
        "remote_root": remote_root,
        "description": "Campaign-level Auto V5 orchestrator smoke control generated deterministically from architecture and HP candidates. Safe plan-only by default.",
        "config_overrides": {"epochs": 20, "strategy": "v5_controller_auto10_smoke20_codegen"},
        "runs": runs,
    }


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


def render_registry_wrapper(arch_name: str) -> str:
    return f'''"""Generated Auto V5 campaign wrapper for {arch_name}."""
import torch
import torch.nn.functional as F

from shared.models import REGISTRY


class Model(torch.nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs):
        super().__init__()
        self.input_adapter = (
            torch.nn.Identity()
            if in_channels == 1
            else torch.nn.Conv2d(in_channels, 1, kernel_size=1)
        )
        self.backbone = REGISTRY.build("{arch_name}", **kwargs)
        self.output_adapter = (
            torch.nn.Identity()
            if out_channels == 1
            else torch.nn.Conv2d(1, out_channels, kernel_size=1)
        )

    def forward(self, x):
        target_hw = x.shape[-2:]
        x = self.input_adapter(x)
        x = self.backbone(x)
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        x = self.output_adapter(x)
        return x
'''


def materialize_codegen_wrappers(local_root: Path, smoke_control: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for row in smoke_control["runs"]:
        arch = str(row["arch_name"])
        model_path = local_root / row["model_file"]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(render_registry_wrapper(arch))
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


def _candidate_from_source_config(path: Path, root: Path) -> dict[str, Any] | None:
    cfg = json.loads(path.read_text())
    arch = cfg.get("arch_name")
    if not arch:
        return None
    run_id = path.parent.name
    campaign = path.parents[2].name
    slug = slugify(str(arch))
    model_file = f"generated_models/v5_controller_auto10_001/{slug}.py"
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


def load_default_candidates(root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in sorted((root / "campaigns" / "v5_ai_curated_001" / "runs").glob("*/train_config.json")):
        row = _candidate_from_source_config(path, root)
        if row is not None:
            candidates.append(row)
    if not candidates:
        raise SystemExit("no default candidates found")
    return candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-root", type=Path, default=_REPO_ROOT)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--campaign", default="v5_controller_auto10_001_smoke20")
    p.add_argument("--run-prefix", default="r_auto10")
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--dry-run", action="store_true", help="plan only; never submit")
    p.add_argument("--execute-codegen", action="store_true", help="write deterministic generated wrapper files for selected architectures")
    p.add_argument("--materialize", action="store_true", help="reserved explicit opt-in for lower-level materialization")
    p.add_argument("--submit-smoke", action="store_true", help="reserved explicit opt-in for smoke submit")
    p.add_argument("--live-crc", action="store_true", help="reserved explicit opt-in for live CRC monitoring")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    candidates = load_default_candidates(args.local_root)
    control = build_smoke_control_from_candidates(
        candidates,
        campaign=args.campaign,
        run_prefix=args.run_prefix,
        count=args.count,
        exclude_arches=DEFAULT_EXCLUDE_ARCHES,
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
        "runs": [row["run_id"] for row in control["runs"]],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
