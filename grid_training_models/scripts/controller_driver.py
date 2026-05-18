#!/usr/bin/env python3
"""Safe outer driver for Auto V5 controller orchestration.

The driver composes existing monitor, promotion, launcher, and retry helpers. It
is intentionally safe by default: it writes local control/plan artifacts, but it
does not submit Condor jobs or perform live CRC actions unless explicit live and
submit flags are provided.
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

from scripts.campaign_orchestrator import (
    build_benchmark_control_from_smoke_passes,
    merge_retry_passes_for_promotion,
)
from scripts.controller_repair_executor import (
    attach_submit_results_to_plan,
    execute_staged_repair_steps,
    materialize_repair_run,
    planned_repair_runs,
    stage_repair_runs,
)
from scripts.controller_retry_executor import materialize_retry_run, planned_retry_runs
from scripts.launch_smoke_from_control import build_plan, load_control, submit_runs, submit_runs_remote_batch
from scripts.controller_state_machine import decide_next_step, write_final_ranking


PARAM_LIMIT = 150_000_000


def merge_repair_passes_for_promotion(
    smoke_control: dict[str, Any],
    controller_state: dict[str, Any],
    repair_states: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a promotion view where original REPAIR rows use passing repair attempts."""
    merged = {"runs": dict(controller_state.get("runs", {}))}
    repair_runs: dict[str, Any] = {}
    for state in repair_states:
        repair_runs.update(state.get("runs", {}))

    for row in smoke_control.get("runs", []):
        run_id = row["run_id"]
        state_row = merged["runs"].get(run_id, {})
        if state_row.get("state_key") == "PASS:RECORD_RESULT":
            continue
        repair_id = str((state_row.get("plan") or {}).get("new_run_id") or f"{run_id}_repair1")
        repair_row = repair_runs.get(repair_id)
        if repair_row and repair_row.get("state_key") == "PASS:RECORD_RESULT":
            promoted = dict(state_row)
            for key in ("classification", "last_evidence_summary", "plan", "cluster_id", "current_run_id"):
                if key in repair_row:
                    promoted[key] = repair_row[key]
            promoted["state_key"] = "PASS:RECORD_RESULT"
            promoted["current_run_id"] = str(repair_row.get("current_run_id") or repair_id)
            promoted["promotion_source"] = "repair_state"
            promoted["promotion_repair_run_id"] = repair_id
            merged["runs"][run_id] = promoted
    return merged


def _param_count_from_state_row(row: dict[str, Any]) -> int | None:
    evidence = row.get("last_evidence_summary") if isinstance(row.get("last_evidence_summary"), dict) else {}
    for key in ("params", "param_count", "parameter_count", "num_params"):
        value = evidence.get(key) if isinstance(evidence, dict) else None
        if value is None:
            value = row.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def filter_param_cap_benchmark_promotion(
    smoke_control: dict[str, Any],
    merged_state: dict[str, Any],
    *,
    limit: int = PARAM_LIMIT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drop smoke PASS rows that violate the formal benchmark parameter cap."""
    states = merged_state.get("runs", {}) if isinstance(merged_state.get("runs"), dict) else {}
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in smoke_control.get("runs", []):
        if not isinstance(row, dict):
            continue
        state_row = states.get(row.get("run_id"), {})
        params = _param_count_from_state_row(state_row) if isinstance(state_row, dict) else None
        if params is not None and params > limit:
            skipped.append({
                "run_id": row.get("run_id"),
                "current_run_id": state_row.get("current_run_id") if isinstance(state_row, dict) else None,
                "state_key": state_row.get("state_key") if isinstance(state_row, dict) else None,
                "params": params,
                "limit": limit,
                "reason": "exceeds formal benchmark parameter cap after smoke-stage retry/repair/pass",
            })
            continue
        kept.append(row)
    if not skipped:
        return smoke_control, []
    if not kept:
        raise SystemExit(f"all smoke-passed runs exceed benchmark parameter cap {limit}")
    filtered = dict(smoke_control)
    filtered["runs"] = kept
    return filtered, skipped


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def smoke_to_benchmark(args: argparse.Namespace) -> dict[str, Path]:
    report_dir = args.report_dir
    smoke_control = load_json(report_dir / "control_smoke20.json")
    smoke_state = load_json(report_dir / "controller_state.smoke20.json")
    retry_states: list[dict[str, Any]] = []
    retry_state_path = report_dir / "controller_state.smoke20.retry1.json"
    if retry_state_path.exists():
        retry_states.append(load_json(retry_state_path))
    repair_states: list[dict[str, Any]] = []
    repair_state_path = report_dir / "controller_state.smoke20.repair1.json"
    if repair_state_path.exists():
        repair_states.append(load_json(repair_state_path))

    merged = merge_retry_passes_for_promotion(smoke_control, smoke_state, retry_states)
    merged = merge_repair_passes_for_promotion(smoke_control, merged, repair_states)
    merged_path = report_dir / "controller_state.smoke20.promotion_merged.json"
    write_json(merged_path, merged)

    benchmark_campaign = args.benchmark_campaign or str(smoke_control.get("campaign", "")).replace("_smoke20", "_benchmark200")
    if not benchmark_campaign or benchmark_campaign == str(smoke_control.get("campaign", "")):
        raise SystemExit("--benchmark-campaign is required when smoke campaign name does not end with _smoke20")
    promotion_control = smoke_control
    if args.allow_partial_promotion:
        states = merged.get("runs", {})
        attempt_states: dict[str, Any] = {}
        for attempt_state in retry_states + repair_states:
            attempt_states.update(attempt_state.get("runs", {}))
        passed_runs = []
        skipped_runs = []
        for row in smoke_control.get("runs", []):
            state_row = states.get(row["run_id"], {})
            if state_row.get("state_key") == "PASS:RECORD_RESULT":
                passed_runs.append(row)
            else:
                final_row = state_row
                planned_id = str((state_row.get("plan") or {}).get("new_run_id") or "")
                if planned_id and planned_id in attempt_states:
                    final_row = attempt_states[planned_id]
                skipped_runs.append({
                    "run_id": row["run_id"],
                    "state_key": final_row.get("state_key"),
                    "current_run_id": final_row.get("current_run_id"),
                })
        if not passed_runs:
            raise SystemExit("partial promotion requested but no smoke runs passed")
        promotion_control = dict(smoke_control)
        promotion_control["runs"] = passed_runs
        write_json(report_dir / "partial_promotion_skipped.smoke20.json", {"skipped_runs": skipped_runs})
    promotion_control, param_cap_skipped = filter_param_cap_benchmark_promotion(promotion_control, merged)
    if param_cap_skipped:
        write_json(report_dir / "benchmark_param_cap_skipped.smoke20.json", {"skipped_runs": param_cap_skipped})
    run_prefix = args.benchmark_run_prefix or "r_benchmark"
    benchmark_control = build_benchmark_control_from_smoke_passes(
        promotion_control,
        merged,
        campaign=benchmark_campaign,
        run_prefix=run_prefix,
    )
    control_path = report_dir / "control_benchmark200.json"
    write_json(control_path, benchmark_control)

    plan = build_plan(args.local_root, benchmark_control, materialize=args.materialize_benchmark)
    plan_path = report_dir / "launch_plan_benchmark200.json"
    write_json(plan_path, plan)
    return {"merged_state": merged_path, "benchmark_control": control_path, "benchmark_plan": plan_path}


def validate_benchmark_control_smoke_gate(control: dict[str, Any]) -> None:
    """Reject benchmark controls that source curated configs directly.

    Benchmark200 is a performance screen, not the first runtime gate. Standard
    Auto V5 flow must be curated/source configs -> smoke20 -> monitor/retry/repair
    -> promotion merge -> benchmark200. Therefore every benchmark run should
    source a smoke campaign/run, not `v5_ai_curated_001` or another non-smoke
    campaign directly.
    """
    if str(control.get("stage") or "") != "benchmark200":
        return
    bad: list[str] = []
    for row in control.get("runs", []):
        if not isinstance(row, dict):
            continue
        source_campaign = str(row.get("source_campaign") or "")
        source_run_id = str(row.get("source_run_id") or "")
        if "smoke20" not in source_campaign and "smoke20" not in source_run_id:
            bad.append(f"{row.get('run_id')} <- {source_campaign}/{source_run_id}")
    if bad:
        preview = "; ".join(bad[:5])
        extra = "" if len(bad) <= 5 else f"; ... +{len(bad) - 5} more"
        raise SystemExit(
            "benchmark200 control failed smoke-gate validation: "
            "runs must source smoke20 promoted attempts, not curated configs directly. "
            f"Examples: {preview}{extra}"
        )


def submit_benchmark(args: argparse.Namespace) -> Path:
    if args.submit_benchmark and not args.live_crc and not args.dry_run:
        raise SystemExit("--submit-benchmark requires --live-crc unless --dry-run is set")
    control = load_control(args.report_dir / "control_benchmark200.json")
    validate_benchmark_control_smoke_gate(control)
    plan = build_plan(args.local_root, control, materialize=True)
    if args.submit_benchmark and args.live_crc and not args.dry_run:
        plan["submit_results"] = submit_runs_remote_batch(args.local_root, plan)
        write_json(args.report_dir / "benchmark_cluster_map.json", attach_submit_results_to_plan(plan))
    output = args.report_dir / "launch_plan_benchmark200.submitted.json"
    write_json(output, plan)
    return output


def _load_optional_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _submitted_run_ids_from_retry_artifacts(*, report_dir: Path, stage: str) -> set[str]:
    submitted: set[str] = set()
    cluster_map = _load_optional_json(report_dir / f"retry_cluster_map.{stage}.json", {})
    if isinstance(cluster_map, dict):
        submitted.update(str(run_id) for run_id in cluster_map.keys())
    retry_plan = _load_optional_json(report_dir / f"retry_submit_plan.{stage}.json", {})
    if isinstance(retry_plan, dict):
        for row in retry_plan.get("runs", []):
            if isinstance(row, dict) and row.get("cluster_id") and isinstance(row.get("run_id"), str):
                submitted.add(str(row["run_id"]))
        for row in retry_plan.get("submit_results", []):
            if isinstance(row, dict) and row.get("cluster_id") and isinstance(row.get("run_id"), str):
                submitted.add(str(row["run_id"]))
    return submitted


def _existing_retry_cluster_map(*, report_dir: Path, stage: str) -> dict[str, str]:
    cluster_map: dict[str, str] = {}
    raw_map = _load_optional_json(report_dir / f"retry_cluster_map.{stage}.json", {})
    if isinstance(raw_map, dict):
        for run_id, cluster_id in raw_map.items():
            if cluster_id:
                cluster_map[str(run_id)] = str(cluster_id)
    retry_plan = _load_optional_json(report_dir / f"retry_submit_plan.{stage}.json", {})
    if isinstance(retry_plan, dict):
        for row in retry_plan.get("runs", []):
            if isinstance(row, dict) and row.get("run_id") and row.get("cluster_id"):
                cluster_map.setdefault(str(row["run_id"]), str(row["cluster_id"]))
        for row in retry_plan.get("submit_results", []):
            if isinstance(row, dict) and row.get("run_id") and row.get("cluster_id"):
                cluster_map.setdefault(str(row["run_id"]), str(row["cluster_id"]))
    return cluster_map


def _existing_retry_submit_results(*, report_dir: Path, stage: str) -> list[dict[str, Any]]:
    retry_plan = _load_optional_json(report_dir / f"retry_submit_plan.{stage}.json", {})
    if not isinstance(retry_plan, dict):
        return []
    results = retry_plan.get("submit_results")
    if not isinstance(results, list):
        return []
    return [dict(row) for row in results if isinstance(row, dict) and isinstance(row.get("run_id"), str)]


def _merge_submit_results(existing: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_run: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in [*existing, *new]:
        run_id = row.get("run_id") if isinstance(row, dict) else None
        if not isinstance(run_id, str):
            continue
        if run_id not in merged_by_run:
            order.append(run_id)
        merged_by_run[run_id] = dict(row)
    return [merged_by_run[run_id] for run_id in order]


def _attach_cluster_ids_to_plan(plan: dict[str, Any], cluster_map: dict[str, str]) -> None:
    for row in plan.get("runs", []):
        if isinstance(row, dict) and isinstance(row.get("run_id"), str) and row["run_id"] in cluster_map:
            row["cluster_id"] = cluster_map[row["run_id"]]


def _stage_retry(args: argparse.Namespace, *, stage: str) -> dict[str, Path]:
    report_dir = args.report_dir
    control = load_json(report_dir / f"control_{stage}.json")
    launch_plan = load_json(report_dir / f"launch_plan_{stage}.submitted.json")
    state = load_json(report_dir / f"controller_state.{stage}.json")
    campaign = str(control.get("campaign") or launch_plan.get("campaign"))
    retry_runs = planned_retry_runs(control=control, launch_plan=launch_plan, state=state)
    retry_control = {
        "campaign": campaign,
        "remote_root": control.get("remote_root") or launch_plan.get("remote_root"),
        "stage": control.get("stage") or launch_plan.get("stage") or stage,
        "runs": retry_runs,
    }
    retry_control_path = report_dir / f"retry_control.{stage}.json"
    write_json(retry_control_path, retry_control)

    already_submitted = _submitted_run_ids_from_retry_artifacts(report_dir=report_dir, stage=stage)
    missing_retry_runs = [run for run in retry_runs if str(run.get("run_id")) not in already_submitted]

    if args.materialize_retry:
        remote_root = str(retry_control.get("remote_root") or "") or None
        for run in missing_retry_runs:
            materialize_retry_run(local_root=args.local_root, campaign=campaign, run=run, remote_root=remote_root)

    retry_plan = build_plan(args.local_root, retry_control, materialize=False)
    existing_cluster_map = _existing_retry_cluster_map(report_dir=report_dir, stage=stage)
    new_submit_results: list[dict[str, Any]] = []
    new_cluster_map: dict[str, str] = {}
    if args.submit_retry and not args.live_crc and not args.dry_run:
        raise SystemExit("--submit-retry requires --live-crc unless --dry-run is set")
    if args.submit_retry and args.live_crc and not args.dry_run and missing_retry_runs:
        missing_control = {**retry_control, "runs": missing_retry_runs}
        missing_plan = build_plan(args.local_root, missing_control, materialize=False)
        new_submit_results = submit_runs_remote_batch(args.local_root, missing_plan)
        missing_plan["submit_results"] = new_submit_results
        new_cluster_map = attach_submit_results_to_plan(missing_plan)

    merged_cluster_map = {**existing_cluster_map, **new_cluster_map}
    _attach_cluster_ids_to_plan(retry_plan, merged_cluster_map)
    merged_submit_results = _merge_submit_results(
        _existing_retry_submit_results(report_dir=report_dir, stage=stage),
        new_submit_results,
    )
    if merged_submit_results:
        retry_plan["submit_results"] = merged_submit_results
    if merged_cluster_map:
        write_json(report_dir / f"retry_cluster_map.{stage}.json", merged_cluster_map)
    retry_plan_path = report_dir / f"retry_submit_plan.{stage}.json"
    write_json(retry_plan_path, retry_plan)
    return {"retry_control": retry_control_path, "retry_plan": retry_plan_path}


def _stage_retry_from_artifacts(
    args: argparse.Namespace,
    *,
    control_path: Path,
    launch_plan_path: Path,
    state_path: Path,
    output_suffix: str,
) -> dict[str, Path]:
    report_dir = args.report_dir
    control = load_json(control_path)
    launch_plan = load_json(launch_plan_path)
    state = load_json(state_path)
    campaign = str(control.get("campaign") or launch_plan.get("campaign"))
    retry_runs = planned_retry_runs(control=control, launch_plan=launch_plan, state=state)
    retry_control = {
        "campaign": campaign,
        "remote_root": control.get("remote_root") or launch_plan.get("remote_root"),
        "stage": control.get("stage") or launch_plan.get("stage") or "smoke20",
        "runs": retry_runs,
    }
    retry_control_path = report_dir / f"repair_retry_control.{output_suffix}.json"
    write_json(retry_control_path, retry_control)

    if args.materialize_retry:
        remote_root = str(retry_control.get("remote_root") or "") or None
        for run in retry_runs:
            materialize_retry_run(local_root=args.local_root, campaign=campaign, run=run, remote_root=remote_root)

    retry_plan = build_plan(args.local_root, retry_control, materialize=False)
    if args.submit_retry and not args.live_crc and not args.dry_run:
        raise SystemExit("--submit-retry requires --live-crc unless --dry-run is set")
    if args.submit_retry and args.live_crc and not args.dry_run:
        retry_plan["submit_results"] = submit_runs_remote_batch(args.local_root, retry_plan)
        write_json(report_dir / f"repair_retry_cluster_map.{output_suffix}.json", attach_submit_results_to_plan(retry_plan))
    retry_plan_path = report_dir / f"repair_retry_submit_plan.{output_suffix}.json"
    write_json(retry_plan_path, retry_plan)
    return {"retry_control": retry_control_path, "retry_plan": retry_plan_path}


def smoke_repair_retry(args: argparse.Namespace) -> dict[str, Path]:
    return _stage_retry_from_artifacts(
        args,
        control_path=args.report_dir / "repair_control.smoke20.json",
        launch_plan_path=args.report_dir / "repair_submit_plan.smoke20.json",
        state_path=args.report_dir / "controller_state.smoke20.repair1.json",
        output_suffix="smoke20.repair1",
    )


def _stage_repair_from_artifacts(
    args: argparse.Namespace,
    *,
    control_path: Path,
    launch_plan_path: Path,
    state_path: Path,
    output_suffix: str,
) -> dict[str, Path]:
    report_dir = args.report_dir
    control = load_json(control_path)
    launch_plan = load_json(launch_plan_path)
    state = load_json(state_path)
    campaign = str(control.get("campaign") or launch_plan.get("campaign") or state.get("campaign"))
    remote_root = str(control.get("remote_root") or launch_plan.get("remote_root") or state.get("remote_root") or "")
    if not campaign or not remote_root:
        raise SystemExit("control, launch plan, or state must define campaign and remote_root for repair-after-repair")
    if args.submit_repair and not args.live_crc and not args.dry_run:
        raise SystemExit("--submit-repair requires --live-crc unless --dry-run is set")
    if args.submit_repair and not args.materialize_repair and not args.dry_run:
        raise SystemExit("--submit-repair requires --materialize-repair so repair configs are validated before submission")
    if args.execute_repair and not args.stage_repair:
        raise SystemExit("--execute-repair requires --stage-repair so repair commands are explicit and inspectable")

    repair_runs = planned_repair_runs(control=control, launch_plan=launch_plan, state=state)
    repair_control = {
        "campaign": campaign,
        "remote_root": remote_root,
        "stage": control.get("stage") or launch_plan.get("stage") or state.get("stage") or "smoke20",
        "runs": repair_runs,
    }
    repair_control_path = report_dir / f"repair_repair_control.{output_suffix}.json"
    write_json(repair_control_path, repair_control)

    if args.stage_repair:
        stage_repair_runs(control=control, state=state, runs=repair_runs, output_root=report_dir / "repairs", local_root=args.local_root)
    if args.execute_repair:
        execution_log = {"runs": execute_staged_repair_steps(repair_output_root=report_dir / "repairs", runs=repair_runs, cwd=args.local_root, dry_run=args.dry_run)}
        write_json(report_dir / f"repair_repair_execution_log.{output_suffix}.json", execution_log)
    if args.materialize_repair:
        for run in repair_runs:
            materialize_repair_run(local_root=args.local_root, campaign=campaign, remote_root=remote_root, run=run)

    repair_plan = build_plan(args.local_root, repair_control, materialize=False)
    if args.submit_repair and args.live_crc and not args.dry_run:
        repair_plan["submit_results"] = submit_runs_remote_batch(args.local_root, repair_plan)
        write_json(report_dir / f"repair_repair_cluster_map.{output_suffix}.json", attach_submit_results_to_plan(repair_plan))
    repair_plan_path = report_dir / f"repair_repair_submit_plan.{output_suffix}.json"
    write_json(repair_plan_path, repair_plan)
    return {"repair_control": repair_control_path, "repair_plan": repair_plan_path}


def smoke_repair_repair(args: argparse.Namespace) -> dict[str, Path]:
    return _stage_repair_from_artifacts(
        args,
        control_path=args.report_dir / "repair_control.smoke20.json",
        launch_plan_path=args.report_dir / "repair_submit_plan.smoke20.json",
        state_path=args.report_dir / "controller_state.smoke20.repair1.json",
        output_suffix="smoke20.repair1",
    )


def smoke_retry(args: argparse.Namespace) -> dict[str, Path]:
    return _stage_retry(args, stage="smoke20")


def benchmark_retry(args: argparse.Namespace) -> dict[str, Path]:
    return _stage_retry(args, stage="benchmark200")


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def state_has_passing_attempt(states: list[dict[str, Any]], run_id: str) -> bool:
    for state in states:
        row = (state.get("runs") or {}).get(run_id)
        if row and row.get("state_key") == "PASS:RECORD_RESULT":
            return True
    return False


def smoke_repair(args: argparse.Namespace) -> dict[str, Path]:
    """Orchestrator-owned bridge from smoke REPAIR state to repair attempts.

    This keeps monitor-only code read-only while allowing the outer controller to
    advance planned repair attempts under explicit gates. Default/dry-run writes
    only local repair control and plan artifacts. Live Condor submission requires
    both --submit-repair and --live-crc.
    """
    report_dir = args.report_dir
    control = load_json(report_dir / "control_smoke20.json")
    launch_plan = load_json(report_dir / "launch_plan_smoke20.submitted.json")
    state = load_json(report_dir / "controller_state.smoke20.json")
    campaign = str(control.get("campaign") or launch_plan.get("campaign") or state.get("campaign"))
    remote_root = str(control.get("remote_root") or launch_plan.get("remote_root") or state.get("remote_root") or "")
    if not campaign or not remote_root:
        raise SystemExit("control, launch plan, or state must define campaign and remote_root for smoke repair")

    if args.submit_repair and not args.live_crc and not args.dry_run:
        raise SystemExit("--submit-repair requires --live-crc unless --dry-run is set")
    if args.submit_repair and not args.materialize_repair and not args.dry_run:
        raise SystemExit("--submit-repair requires --materialize-repair so repair configs are validated before submission")
    if args.execute_repair and not args.stage_repair:
        raise SystemExit("--execute-repair requires --stage-repair so repair commands are explicit and inspectable")

    repair_runs = planned_repair_runs(control=control, launch_plan=launch_plan, state=state)
    repair_control = {
        "campaign": campaign,
        "remote_root": remote_root,
        "stage": control.get("stage") or launch_plan.get("stage") or state.get("stage") or "smoke20",
        "runs": repair_runs,
    }
    repair_control_path = report_dir / "repair_control.smoke20.json"
    write_json(repair_control_path, repair_control)

    if args.stage_repair:
        stage_repair_runs(
            control=control,
            state=state,
            runs=repair_runs,
            output_root=report_dir / "repairs",
            local_root=args.local_root,
        )
    if args.execute_repair:
        execution_log = {"runs": execute_staged_repair_steps(repair_output_root=report_dir / "repairs", runs=repair_runs, cwd=args.local_root, dry_run=args.dry_run)}
        write_json(report_dir / "repair_execution_log.smoke20.json", execution_log)

    if args.materialize_repair:
        for run in repair_runs:
            materialize_repair_run(local_root=args.local_root, campaign=campaign, remote_root=remote_root, run=run)

    repair_plan = build_plan(args.local_root, repair_control, materialize=False)
    if args.submit_repair and args.live_crc and not args.dry_run:
        repair_plan["submit_results"] = submit_runs_remote_batch(args.local_root, repair_plan)
        cluster_map = attach_submit_results_to_plan(repair_plan)
        write_json(report_dir / "repair_cluster_map.smoke20.json", cluster_map)

    repair_plan_path = report_dir / "repair_submit_plan.smoke20.json"
    write_json(repair_plan_path, repair_plan)
    return {"repair_control": repair_control_path, "repair_plan": repair_plan_path}


def auto_advance(args: argparse.Namespace) -> dict[str, Path]:
    """Select and execute the next safe controller-driver step.

    The top-level loop remains conservative: it either calls an existing safe
    driver step or writes a local decision artifact. Live Condor side effects are
    still gated by the existing explicit submit flags plus --live-crc.
    """
    report_dir = args.report_dir
    decision_path = report_dir / "auto_advance_plan.json"
    generic_decision = decide_next_step(report_dir)
    if generic_decision.get("decision") == "complete":
        decision = {**generic_decision, "dry_run": bool(args.dry_run), "live_crc": bool(args.live_crc)}
        write_json(decision_path, decision)
        return {"auto_advance_plan": decision_path}
    if generic_decision.get("decision") == "final-ranking":
        ranking = write_final_ranking(report_dir)
        decision = {**generic_decision, "dry_run": bool(args.dry_run), "live_crc": bool(args.live_crc), "best": ranking["ranking_by_r2_median"][0]}
        write_json(decision_path, decision)
        return {"auto_advance_plan": decision_path, "final_ranking": report_dir / "final_ranking.json"}

    if args.submit_benchmark:
        submitted_plan = submit_benchmark(args)
        decision = {
            "decision": "submit-benchmark",
            "dry_run": bool(args.dry_run),
            "live_crc": bool(args.live_crc),
            "outputs": {"submitted_plan": str(submitted_plan)},
        }
        write_json(decision_path, decision)
        return {"auto_advance_plan": decision_path, "submitted_plan": submitted_plan}

    benchmark_state = load_json_if_exists(report_dir / "controller_state.benchmark200.json")
    benchmark_control_path = report_dir / "control_benchmark200.json"
    benchmark_retry_states = [state for state in [load_json_if_exists(report_dir / "controller_state.benchmark200.retry1.json")] if state]
    if benchmark_state and benchmark_control_path.exists():
        for run_id, row in (benchmark_state.get("runs") or {}).items():
            plan = row.get("plan") or {}
            action = str(plan.get("action") or "").upper()
            state_key = str(row.get("state_key") or "")
            if action == "RETRY" or state_key.endswith(":RETRY"):
                retry_id = str(plan.get("new_run_id") or f"{run_id}_retry1")
                if not state_has_passing_attempt(benchmark_retry_states, retry_id):
                    outputs = benchmark_retry(args)
                    decision = {
                        "decision": "benchmark-retry",
                        "dry_run": bool(args.dry_run),
                        "live_crc": bool(args.live_crc),
                        "reason": f"{run_id} requires benchmark retry attempt {retry_id}",
                        "outputs": {key: str(value) for key, value in outputs.items()},
                    }
                    write_json(decision_path, decision)
                    return {"auto_advance_plan": decision_path, **outputs}

    smoke_state = load_json_if_exists(report_dir / "controller_state.smoke20.json")
    smoke_control_path = report_dir / "control_smoke20.json"
    retry_states = [state for state in [load_json_if_exists(report_dir / "controller_state.smoke20.retry1.json")] if state]
    repair_states = [state for state in [load_json_if_exists(report_dir / "controller_state.smoke20.repair1.json")] if state]

    if smoke_state and smoke_control_path.exists():
        for run_id, row in (smoke_state.get("runs") or {}).items():
            plan = row.get("plan") or {}
            action = str(plan.get("action") or "").upper()
            state_key = str(row.get("state_key") or "")
            new_run_id = str(plan.get("new_run_id") or "")
            if action == "REPAIR" or state_key.endswith(":REPAIR"):
                repair_id = new_run_id or f"{run_id}_repair1"
                if not state_has_passing_attempt(repair_states, repair_id):
                    outputs = smoke_repair(args)
                    decision = {
                        "decision": "smoke-repair",
                        "dry_run": bool(args.dry_run),
                        "live_crc": bool(args.live_crc),
                        "reason": f"{run_id} requires repair attempt {repair_id}",
                        "outputs": {key: str(value) for key, value in outputs.items()},
                    }
                    write_json(decision_path, decision)
                    return {"auto_advance_plan": decision_path, **outputs}
            if action == "RETRY" or state_key.endswith(":RETRY"):
                retry_id = new_run_id or f"{run_id}_retry1"
                if not state_has_passing_attempt(retry_states, retry_id):
                    decision = {
                        "decision": "smoke-retry",
                        "dry_run": bool(args.dry_run),
                        "live_crc": bool(args.live_crc),
                        "reason": f"{run_id} requires retry attempt {retry_id}",
                        "run_id": run_id,
                        "retry_run_id": retry_id,
                        "outputs": {key: str(value) for key, value in smoke_retry(args).items()},
                    }
                    write_json(decision_path, decision)
                    return {"auto_advance_plan": decision_path}

        outputs = smoke_to_benchmark(args)
        decision = {
            "decision": "smoke-to-benchmark",
            "dry_run": bool(args.dry_run),
            "live_crc": bool(args.live_crc),
            "outputs": {key: str(value) for key, value in outputs.items()},
        }
        write_json(decision_path, decision)
        return {"auto_advance_plan": decision_path, **outputs}

    if benchmark_control_path.exists():
        decision = {
            "decision": "benchmark-ready",
            "dry_run": bool(args.dry_run),
            "live_crc": bool(args.live_crc),
            "reason": "benchmark control exists; no unresolved benchmark retry is pending, or pass --submit-benchmark --live-crc to submit",
        }
        write_json(decision_path, decision)
        return {"auto_advance_plan": decision_path}

    raise SystemExit("auto-advance could not find a known controller state to advance")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=("auto-advance", "smoke-to-benchmark", "smoke-retry", "submit-benchmark", "benchmark-retry", "smoke-repair", "smoke-repair-retry", "smoke-repair-repair"), required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Do not submit live CRC jobs. This is the safe default behavior.")
    parser.add_argument("--live-crc", action="store_true", help="Allow live CRC submit/poll actions when paired with explicit submit flags.")
    parser.add_argument("--benchmark-campaign", default=None)
    parser.add_argument("--benchmark-run-prefix", default=None)
    parser.add_argument("--allow-partial-promotion", action="store_true", help="Promote only smoke runs whose final original/retry/repair state is PASS; write skipped runs sidecar.")
    parser.add_argument("--materialize-benchmark", action="store_true")
    parser.add_argument("--submit-benchmark", action="store_true")
    parser.add_argument("--materialize-retry", action="store_true")
    parser.add_argument("--submit-retry", action="store_true")
    parser.add_argument("--stage-repair", action="store_true", help="Stage repair contexts/commands for planned smoke repair attempts.")
    parser.add_argument("--execute-repair", action="store_true", help="Explicit opt-in to execute staged Claude/Codex/Claude repair commands.")
    parser.add_argument("--materialize-repair", action="store_true", help="Write sanitized repair train_config.json files for planned smoke repair attempts.")
    parser.add_argument("--submit-repair", action="store_true", help="Submit planned smoke repair attempts; requires --live-crc unless --dry-run.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.step == "auto-advance":
        outputs = auto_advance(args)
    elif args.step == "smoke-to-benchmark":
        outputs = smoke_to_benchmark(args)
    elif args.step == "smoke-retry":
        outputs = smoke_retry(args)
    elif args.step == "submit-benchmark":
        outputs = {"submitted_plan": submit_benchmark(args)}
    elif args.step == "benchmark-retry":
        outputs = benchmark_retry(args)
    elif args.step == "smoke-repair":
        outputs = smoke_repair(args)
    elif args.step == "smoke-repair-retry":
        outputs = smoke_repair_retry(args)
    elif args.step == "smoke-repair-repair":
        outputs = smoke_repair_repair(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown step: {args.step}")
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
