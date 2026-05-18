#!/usr/bin/env python3
"""Generic artifact-driven state machine for Grid controller rounds.

This module deliberately does not plan a new set of candidates. It only inspects
an existing controller report directory and decides the next generic transition
for any Grid-like round: smoke retry/repair, promotion, benchmark submit,
benchmark retry, final ranking, or complete.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

BASELINE_REFERENCE = {
    "source_run": "<BASELINE_HPC_SOURCE_ROOT>/campaigns/orthogonal exploratory sweep_200ep/baseline",
    "experiment_id": "orthogonal exploratory sweep_200ep_baseline",
    "r2_median": 0.7085034105693335,
    "r2_global": 0.6566194830043587,
    "mae_median": 0.09351099282503128,
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _runs(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = state.get("runs") or {}
    if not isinstance(rows, dict):
        return {}
    return {str(k): v for k, v in rows.items() if isinstance(v, dict)}


def _state_key(row: dict[str, Any]) -> str:
    return str(row.get("state_key") or "")


def _plan_action(row: dict[str, Any]) -> str:
    plan = row.get("plan") or {}
    return str(plan.get("action") or "").upper() if isinstance(plan, dict) else ""


def _planned_id(source_run_id: str, row: dict[str, Any], suffix: str) -> str:
    plan = row.get("plan") or {}
    if isinstance(plan, dict) and plan.get("new_run_id"):
        return str(plan["new_run_id"])
    return f"{source_run_id}_{suffix}"


def _has_recorded_result(state: dict[str, Any], run_id: str) -> bool:
    row = _runs(state).get(run_id)
    return bool(row and _state_key(row).endswith(":RECORD_RESULT"))


def _has_pass(state: dict[str, Any], run_id: str) -> bool:
    return _has_recorded_result(state, run_id)


def _has_unresolved_second_repair(report_dir: Path, stage: str, source_run_id: str, row: dict[str, Any]) -> bool:
    """Return true when repair1 and repair1->repair1 both failed into REPAIR.

    The controller repair budget allows the original repair plus one repair-after-
    repair. If the second repair attempt still asks for REPAIR, the original row
    is exhausted for this clock/state-machine cycle and should not block other
    pending RETRY rows forever.
    """
    repair1_state = load_json(report_dir / f"controller_state.{stage}.repair1.json")
    repair2_state = load_json(report_dir / f"controller_state.{stage}.repair1.repair1.json")
    if not repair1_state or not repair2_state:
        return False
    repair1_id = _planned_id(source_run_id, row, "repair1")
    repair1_row = _runs(repair1_state).get(repair1_id)
    if not repair1_row:
        return False
    repair2_id = _planned_id(repair1_id, repair1_row, "repair1")
    repair2_row = _runs(repair2_state).get(repair2_id)
    if not repair2_row or _state_key(repair2_row) == "PASS:RECORD_RESULT":
        return False
    return _plan_action(repair2_row) == "REPAIR" or _state_key(repair2_row).endswith(":REPAIR")


def _load_attempt_states(report_dir: Path, stage: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for label in ("retry1", "repair1"):
        path = report_dir / f"controller_state.{stage}.{label}.json"
        state = load_json(path)
        if state:
            out.append((label, state))
    return out


def _pending_attempt_for_state(report_dir: Path, stage: str, state: dict[str, Any]) -> dict[str, Any] | None:
    attempts = _load_attempt_states(report_dir, stage)
    retry_state = dict((label, s) for label, s in attempts).get("retry1", {})
    repair_state = dict((label, s) for label, s in attempts).get("repair1", {})
    for run_id, row in _runs(state).items():
        action = _plan_action(row)
        state_key = _state_key(row)
        if action == "REPAIR" or state_key.endswith(":REPAIR"):
            if _has_unresolved_second_repair(report_dir, stage, run_id, row):
                continue
            attempt_id = _planned_id(run_id, row, "repair1")
            if not _has_pass(repair_state, attempt_id):
                return {
                    "decision": f"{stage.replace('20', '')}-repair" if stage == "smoke20" else f"{stage}-repair",
                    "stage": stage,
                    "run_id": run_id,
                    "attempt_run_id": attempt_id,
                    "side_effect": f"controller_driver:{'smoke-repair' if stage == 'smoke20' else stage + '-repair'}",
                }
        if action == "RETRY" or state_key.endswith(":RETRY"):
            attempt_id = _planned_id(run_id, row, "retry1")
            if not _has_pass(retry_state, attempt_id):
                return {
                    "decision": f"{stage.replace('20', '')}-retry" if stage == "smoke20" else f"{stage}-retry",
                    "stage": stage,
                    "run_id": run_id,
                    "attempt_run_id": attempt_id,
                    "side_effect": f"controller_driver:{'smoke-retry' if stage == 'smoke20' else stage + '-retry'}",
                }
    return None


def _all_smoke_resolved_to_pass(report_dir: Path) -> bool:
    control = load_json(report_dir / "control_smoke20.json")
    state = load_json(report_dir / "controller_state.smoke20.json")
    if not control or not state:
        return False
    merged = dict(state)
    merged_runs = _runs(merged)
    for label, attempt_state in _load_attempt_states(report_dir, "smoke20"):
        for run in control.get("runs", []):
            if not isinstance(run, dict) or not run.get("run_id"):
                continue
            run_id = str(run["run_id"])
            row = merged_runs.get(run_id, {})
            if _state_key(row) == "PASS:RECORD_RESULT":
                continue
            suffix = "retry1" if label == "retry1" else "repair1"
            attempt_id = _planned_id(run_id, row, suffix)
            if _has_pass(attempt_state, attempt_id):
                promoted = dict(row)
                promoted["state_key"] = "PASS:RECORD_RESULT"
                promoted["current_run_id"] = attempt_id
                merged_runs[run_id] = promoted
    return all(merged_runs.get(str(run.get("run_id")), {}).get("state_key") == "PASS:RECORD_RESULT" for run in control.get("runs", []) if isinstance(run, dict))


def resolve_final_benchmark_attempts(report_dir: Path) -> list[dict[str, Any]]:
    control = load_json(report_dir / "control_benchmark200.json")
    state = load_json(report_dir / "controller_state.benchmark200.json")
    retry_state = load_json(report_dir / "controller_state.benchmark200.retry1.json")
    repair_state = load_json(report_dir / "controller_state.benchmark200.repair1.json")
    if not control or not state:
        return []
    final: list[dict[str, Any]] = []
    state_runs = _runs(state)
    for row in control.get("runs", []):
        if not isinstance(row, dict) or not row.get("run_id"):
            continue
        run_id = str(row["run_id"])
        state_row = state_runs.get(run_id, {})
        if _has_recorded_result(state, run_id):
            final.append({"run_id": str(state_row.get("current_run_id") or run_id), "source_run_id": run_id, "resolution": "original", "control_row": row})
            continue
        retry_id = _planned_id(run_id, state_row, "retry1")
        if _has_pass(retry_state, retry_id):
            final.append({"run_id": retry_id, "source_run_id": run_id, "resolution": "retry1", "control_row": row})
            continue
        repair_id = _planned_id(run_id, state_row, "repair1")
        if _has_pass(repair_state, repair_id):
            final.append({"run_id": repair_id, "source_run_id": run_id, "resolution": "repair1", "control_row": row})
    return final


def decide_next_step(report_dir: Path) -> dict[str, Any]:
    report_dir = Path(report_dir)
    if not report_dir.exists() or not (report_dir / "control_smoke20.json").exists() and not (report_dir / "control_benchmark200.json").exists():
        return {
            "decision": "needs-round-control",
            "reason": "No smoke or benchmark control exists. Create or provide a round control; the generic state machine does not choose candidates or plan a next-10 batch.",
            "side_effect": "none",
        }

    if (report_dir / "final_ranking.json").exists():
        return {"decision": "complete", "reason": "final_ranking.json exists", "side_effect": "none"}

    benchmark_state = load_json(report_dir / "controller_state.benchmark200.json")
    if benchmark_state:
        pending = _pending_attempt_for_state(report_dir, "benchmark200", benchmark_state)
        if pending:
            if pending["decision"] == "benchmark200-retry":
                pending["decision"] = "benchmark-retry"
                pending["side_effect"] = "controller_driver:benchmark-retry"
            return pending
        final_attempts = resolve_final_benchmark_attempts(report_dir)
        control = load_json(report_dir / "control_benchmark200.json")
        expected = len([r for r in control.get("runs", []) if isinstance(r, dict)])
        if expected and len(final_attempts) == expected:
            return {
                "decision": "final-ranking",
                "stage": "benchmark200",
                "final_attempts": [row["run_id"] for row in final_attempts],
                "side_effect": "write-final-ranking",
            }
        return {"decision": "benchmark-monitor", "stage": "benchmark200", "side_effect": "controller_monitor:benchmark200"}

    if (report_dir / "control_benchmark200.json").exists():
        if not (report_dir / "launch_plan_benchmark200.submitted.json").exists():
            return {"decision": "submit-benchmark", "stage": "benchmark200", "side_effect": "controller_driver:submit-benchmark"}
        return {"decision": "benchmark-monitor", "stage": "benchmark200", "side_effect": "controller_monitor:benchmark200"}

    smoke_state = load_json(report_dir / "controller_state.smoke20.json")
    if smoke_state:
        pending = _pending_attempt_for_state(report_dir, "smoke20", smoke_state)
        if pending:
            if pending["decision"] == "smoke-retry":
                pending["side_effect"] = "controller_driver:smoke-retry"
            return pending
        if _all_smoke_resolved_to_pass(report_dir):
            return {"decision": "smoke-to-benchmark", "stage": "smoke20", "side_effect": "controller_driver:smoke-to-benchmark"}
        return {"decision": "smoke-monitor", "stage": "smoke20", "side_effect": "controller_monitor:smoke20"}

    if (report_dir / "launch_plan_smoke20.submitted.json").exists():
        return {"decision": "smoke-monitor", "stage": "smoke20", "side_effect": "controller_monitor:smoke20"}
    return {"decision": "submit-smoke", "stage": "smoke20", "side_effect": "launch_smoke_from_control"}


def _metric_value(metrics: dict[str, Any], key: str) -> Any:
    val = metrics.get("val_metrics") if isinstance(metrics.get("val_metrics"), dict) else metrics.get("val")
    if isinstance(val, dict) and key in val:
        return val[key]
    return metrics.get(key)


def default_local_metrics_loader(report_dir: Path, attempt: dict[str, Any]) -> dict[str, Any]:
    control = load_json(report_dir / "control_benchmark200.json")
    campaign = str(control.get("campaign") or "")
    run_id = str(attempt["run_id"])
    path = report_dir.parents[1] / "campaigns" / campaign / "runs" / run_id / "metrics.json"
    if path.exists():
        return load_json(path)

    host = os.environ.get("CRC_HOST")
    socket = os.environ.get("CRC_CONTROL_PATH")
    remote_root = str(control.get("remote_root") or "")
    if host and socket and remote_root and campaign:
        remote_path = f"{remote_root.rstrip('/')}/campaigns/{campaign}/runs/{run_id}/metrics.json"
        try:
            completed = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    f"ControlPath={socket}",
                    "-o",
                    "ConnectTimeout=120",
                    host,
                    f"cat {shlex.quote(remote_path)}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=90,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise FileNotFoundError(f"metrics not found locally or via CRC for {run_id}: local={path}, remote={remote_path}") from exc
        data = json.loads(completed.stdout)
        if not isinstance(data, dict):
            raise FileNotFoundError(f"remote metrics is not a JSON object for {run_id}: {remote_path}")
        return data

    raise FileNotFoundError(f"metrics not found locally for {run_id}: {path}")


def write_final_ranking(
    report_dir: Path,
    *,
    metrics_loader: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report_dir = Path(report_dir)
    baseline = dict(baseline or BASELINE_REFERENCE)
    attempts = resolve_final_benchmark_attempts(report_dir)
    if not attempts:
        raise SystemExit("no resolved benchmark PASS attempts available for final ranking")
    loader = metrics_loader or (lambda attempt: default_local_metrics_loader(report_dir, attempt))
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        metrics = loader(attempt)
        status = metrics.get("status")
        if status != "ok":
            raise SystemExit(f"{attempt['run_id']}: metrics status is not ok: {status}")
        row = {
            "run_id": attempt["run_id"],
            "source_run_id": attempt["source_run_id"],
            "resolution": attempt["resolution"],
            "r2_median": float(_metric_value(metrics, "r2_median")),
            "r2_global": float(_metric_value(metrics, "r2_global")),
            "mae_median": float(_metric_value(metrics, "mae_median")),
            "mae_mean": float(_metric_value(metrics, "mae_mean")),
        }
        row["delta_vs_baseline_r2_median"] = row["r2_median"] - float(baseline["r2_median"])
        row["delta_vs_baseline_r2_global"] = row["r2_global"] - float(baseline["r2_global"])
        row["delta_vs_baseline_mae_median"] = row["mae_median"] - float(baseline["mae_median"])
        rows.append(row)
    rows.sort(key=lambda r: r["r2_median"], reverse=True)
    for idx, row in enumerate(rows, 1):
        row["rank"] = idx
    result = {
        "campaign": load_json(report_dir / "control_benchmark200.json").get("campaign"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_reference": baseline,
        "ranking_by_r2_median": rows,
    }
    write_json(report_dir / "final_ranking.json", result)
    lines = ["# Grid benchmark200 final ranking", "", f"Generated at: `{result['generated_at']}`", ""]
    lines.append(
        "Baseline reference: "
        f"R2_median={baseline['r2_median']:.6f}, R2_global={baseline['r2_global']:.6f}, MAE_median={baseline['mae_median']:.6f}."
    )
    lines.append("")
    for row in rows:
        lines.append(
            f"{row['rank']}. `{row['run_id']}`: "
            f"R2_median={row['r2_median']:.6f}, R2_global={row['r2_global']:.6f}, "
            f"MAE_median={row['mae_median']:.6f}, Î”R2_median={row['delta_vs_baseline_r2_median']:.6f}"
        )
    best = rows[0]
    lines.append("")
    verdict = "beats" if best["delta_vs_baseline_r2_median"] > 0 else "does not beat"
    lines.append(f"Best by R2_median: `{best['run_id']}` {verdict} baseline on R2_median.")
    (report_dir / "final_ranking.md").write_text("\n".join(lines) + "\n")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--write-decision", action="store_true")
    parser.add_argument("--write-ranking", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.write_ranking:
        result = write_final_ranking(args.report_dir)
        print(json.dumps({"final_ranking": str(args.report_dir / "final_ranking.json"), "best": result["ranking_by_r2_median"][0]}, indent=2))
        return 0
    decision = decide_next_step(args.report_dir)
    if args.write_decision:
        write_json(args.report_dir / "state_machine_decision.json", decision)
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



