#!/usr/bin/env python3
"""Materialize controller-planned Auto V5 retry attempts.

This executor is deliberately opt-in and local-first. By default it only writes a
retry control file that can be inspected. It does not submit Condor jobs, does
not call SSH, and does not modify train_config.json unless --materialize is
provided.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.launch_smoke_from_control import build_plan, submit_runs
from scripts.prepare_retry_run import load_config, prepare_retry_config


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_lookup(launch_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = launch_plan.get("runs", [])
    if not isinstance(runs, list):
        raise SystemExit("launch plan runs must be a list")
    out: dict[str, dict[str, Any]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_id = run.get("run_id")
        if isinstance(run_id, str):
            out[run_id] = run
    return out


def planned_retry_runs(
    *,
    control: dict[str, Any],
    launch_plan: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    campaign = str(control.get("campaign") or launch_plan.get("campaign") or "")
    if not campaign:
        raise SystemExit("control or launch plan must define campaign")
    by_run = run_lookup(launch_plan)
    planned: list[dict[str, Any]] = []
    state_runs = state.get("runs", {})
    if not isinstance(state_runs, dict):
        raise SystemExit("controller state runs must be an object")

    for source_run_id, entry in state_runs.items():
        if not isinstance(entry, dict):
            continue
        plan = entry.get("plan", {})
        if not isinstance(plan, dict) or plan.get("action") != "RETRY":
            continue
        source_run_id = str(entry.get("current_run_id") or source_run_id)
        launch_entry = by_run.get(source_run_id, {})
        new_run_id = str(plan.get("new_run_id") or "")
        if not new_run_id:
            raise SystemExit(f"{source_run_id}: retry plan missing new_run_id")
        retry_entry = {
            "source_campaign": campaign,
            "source_run_id": source_run_id,
            "run_id": new_run_id,
            "model_file": launch_entry.get("model_file"),
            "module_name": launch_entry.get("module_name"),
            "submit_tier": plan.get("tier") or launch_entry.get("submit_tier"),
            "batch_size": plan.get("batch_size") or launch_entry.get("batch_size"),
            "reason": plan.get("classification") or entry.get("classification", {}).get("classification") or "RETRY",
        }
        if launch_entry.get("allow_param_cap_relaxation") or str(retry_entry.get("reason")) == "PARAM_TOO_LARGE_RETRY_H100":
            retry_entry["allow_param_cap_relaxation"] = True
        planned.append(retry_entry)
    return planned


def materialize_retry_run(*, local_root: Path, campaign: str, run: dict[str, Any], remote_root: str | None) -> None:
    source_run_id = str(run["source_run_id"])
    new_run_id = str(run["run_id"])
    source_config = local_root / "campaigns" / campaign / "runs" / source_run_id / "train_config.json"
    output_dir = local_root / "campaigns" / campaign / "runs" / new_run_id
    source_cfg = load_config(source_config)
    retry_cfg = prepare_retry_config(source_cfg, new_run_id=new_run_id, batch_size=run.get("batch_size"))
    if remote_root:
        # Never inherit a stale results_dir from the source config. Earlier retry
        # materialization preserved an auto_v3 results root; retries must write
        # under the controller-selected remote_root.
        retry_cfg["results_dir"] = str(Path(str(remote_root)) / "campaigns" / campaign / "runs" / new_run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(retry_cfg, indent=2) + "\n")
    note = {
        "source_config": str(source_config),
        "source_run_id": source_run_id,
        "new_run_id": new_run_id,
        "classification": run.get("reason"),
        "reason": run.get("reason"),
        "tier": run.get("submit_tier"),
        "batch_size": retry_cfg.get("batch_size"),
        "metadata_rule": "retry bookkeeping is stored outside train_config.json",
    }
    (output_dir / "RETRY_NOTE.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in note.items() if v is not None) + "\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--launch-plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, default=Path("."))
    parser.add_argument("--retry-control-output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Only write the retry control file. This is the default behavior unless --materialize is set.")
    parser.add_argument("--materialize", action="store_true", help="Write sanitized retry train_config.json files and RETRY_NOTE sidecars.")
    parser.add_argument("--submit-retry", action="store_true", help="Explicit opt-in to submit materialized retry runs through crc_codegen_smoke_one.sh.")
    parser.add_argument("--submitted-plan-output", type=Path, default=None, help="Optional path for the launcher plan used for retry submission.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.submit_retry and not args.materialize:
        raise SystemExit("--submit-retry requires --materialize so retry configs are validated before submission")

    control = load_json(args.control)
    launch_plan = load_json(args.launch_plan)
    state = load_json(args.state)
    campaign = str(control.get("campaign") or launch_plan.get("campaign"))
    runs = planned_retry_runs(control=control, launch_plan=launch_plan, state=state)
    retry_control = {
        "campaign": campaign,
        "remote_root": control.get("remote_root") or launch_plan.get("remote_root"),
        "stage": control.get("stage") or launch_plan.get("stage") or "smoke20",
        "runs": runs,
    }
    write_json(args.retry_control_output, retry_control)

    if args.materialize:
        remote_root = str(retry_control.get("remote_root") or "") or None
        for run in runs:
            materialize_retry_run(local_root=args.local_root, campaign=campaign, run=run, remote_root=remote_root)

    submitted_plan: dict[str, Any] | None = None
    if args.submit_retry:
        submitted_plan = build_plan(args.local_root, retry_control, materialize=False)
        if not args.dry_run:
            submitted_plan["submit_results"] = submit_runs(args.local_root, submitted_plan)
        if args.submitted_plan_output:
            write_json(args.submitted_plan_output, submitted_plan)

    print(f"Planned {len(runs)} retry run(s)")
    print(f"Wrote retry control {args.retry_control_output}")
    if args.submit_retry:
        if args.dry_run:
            print("Submit-retry dry-run only; no Condor submit was executed")
        else:
            submitted_count = len(submitted_plan.get("submit_results", [])) if submitted_plan else 0
            print(f"Submitted {submitted_count} retry run(s)")
    elif args.materialize:
        print("Materialized sanitized retry configs")
    else:
        print("Dry-run only; no train_config.json files were written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
