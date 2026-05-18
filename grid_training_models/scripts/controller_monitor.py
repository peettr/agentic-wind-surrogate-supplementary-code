#!/usr/bin/env python3
"""Local monitor-only orchestrator for Auto V5 controller campaigns.

This script is intentionally side-effect limited. It is designed to run from the
local WSL/Hermes workspace as the controller state source of truth. In
monitor-only mode it reads a control file, optional launch plan, local run
artifacts, and optional read-only CRC/Condor evidence behind --live-crc, then
writes local controller state and transition events. It does not submit, remove,
retry, repair, or edit train_config.json.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.auto_v5_controller import (
    ACTION_AUTO_FAIL,
    ACTION_RECORD_RESULT,
    ACTION_REPAIR,
    ACTION_RETRY,
    append_event,
    get_base_entry,
    plan_next_step,
    run_state_key,
    should_emit_transition,
)
from scripts.classify_run_failure import classify_evidence, collect_local_evidence


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_paths(remote_root: str, campaign: str, run_id: str, log_root: str) -> tuple[Path, Path]:
    run_dir = Path(remote_root) / "campaigns" / campaign / "runs" / run_id
    log_dir = Path(log_root) / run_id
    return run_dir, log_dir


def summarize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    metrics = evidence.get("metrics") or {}
    summary = {
        "finished": bool(evidence.get("finished")),
        "failed": bool(evidence.get("failed")),
        "metrics_status": metrics.get("status") if isinstance(metrics, dict) else None,
        "heartbeat_epoch": evidence.get("heartbeat_epoch"),
        "latest_train_epoch": evidence.get("latest_train_epoch"),
        "checkpoint_exists": evidence.get("checkpoint_exists"),
        "condor_job_status": evidence.get("condor_job_status"),
    }
    for key in ("error_message", "traceback"):
        if isinstance(metrics, dict) and metrics.get(key):
            summary[key] = metrics.get(key)
    for key in ("params", "shape_ok", "tier", "batch_size", "idle_seconds", "qdate"):
        if evidence.get(key) is not None:
            summary[key] = evidence.get(key)
    return summary


def _manifest_entry_from_plan_row(run_id: str, plan_row: dict[str, Any] | None) -> dict[str, Any]:
    plan_row = plan_row or {}
    total_attempts = int(plan_row.get("total_attempts", 1) or 1)
    retry_count = int(plan_row.get("retry_count", 0) or 0)
    repair_count = int(plan_row.get("repair_count", 0) or 0)
    attempts = [{"run_id": run_id, "type": "initial"}]
    attempts.extend({"run_id": f"{run_id}#retry{i}", "type": "retry"} for i in range(retry_count))
    while len(attempts) < total_attempts:
        attempts.append({"run_id": f"{run_id}#attempt{len(attempts) + 1}", "type": "derived"})
    return {
        "base_run_id": run_id,
        "attempts": attempts[:total_attempts],
        "repairs": [{"run_id": f"{run_id}#repair{i + 1}"} for i in range(repair_count)],
    }


def collect_and_classify_run(
    *,
    remote_root: str,
    campaign: str,
    run_id: str,
    log_root: str,
    condor_fields: dict[str, Any] | None = None,
    evidence_override: dict[str, Any] | None = None,
    plan_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir, log_dir = build_run_paths(remote_root, campaign, run_id, log_root)
    evidence = dict(evidence_override) if evidence_override is not None else collect_local_evidence(run_dir, log_dir)
    if condor_fields:
        evidence.update(condor_fields)
    cfg = evidence.get("train_config")
    if isinstance(cfg, dict):
        evidence.setdefault("arch_name", cfg.get("arch_name"))
        evidence.setdefault("batch_size", cfg.get("batch_size"))
        evidence.setdefault("script_path", cfg.get("script_path"))
    evidence.setdefault("run_id", run_id)
    classification = classify_evidence(evidence).to_dict()
    manifest_entry = _manifest_entry_from_plan_row(run_id, plan_row)
    plan = plan_next_step(manifest_entry, classification)
    state_key = run_state_key(classification, plan)
    return {
        "run_id": run_id,
        "classification": classification,
        "plan": plan,
        "state_key": state_key,
        "evidence_summary": summarize_evidence(evidence),
        "run_dir": str(run_dir),
        "log_dir": str(log_dir),
    }


def _runs_by_id(plan_or_control: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = plan_or_control.get("runs") or []
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("run_id"):
            result[str(row["run_id"])] = dict(row)
    return result


def _seed_state(control: dict[str, Any], launch_plan: dict[str, Any], previous_state: dict[str, Any], now: str) -> dict[str, Any]:
    campaign = str(control.get("campaign") or launch_plan.get("campaign") or previous_state.get("campaign") or "")
    remote_root = str(control.get("remote_root") or launch_plan.get("remote_root") or previous_state.get("remote_root") or "")
    stage = str(control.get("stage") or launch_plan.get("stage") or previous_state.get("stage") or "")
    log_root = str(control.get("log_root") or previous_state.get("log_root") or f"/users/lhu1/condor_v5_logs/{campaign}")
    state = dict(previous_state)
    state.update({
        "campaign": campaign,
        "remote_root": remote_root,
        "log_root": log_root,
        "stage": stage,
        "updated_at": now,
    })
    state.setdefault("created_at", now)
    state.setdefault("runs", {})
    return state


def _run_ids(control: dict[str, Any], launch_plan: dict[str, Any]) -> list[str]:
    ids = list(_runs_by_id(control))
    for run_id in _runs_by_id(launch_plan):
        if run_id not in ids:
            ids.append(run_id)
    return ids


def _cluster_ids_from_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    data = load_json(path)
    if "runs" in data:
        return {str(row["run_id"]): str(row["cluster_id"]) for row in data.get("runs", []) if isinstance(row, dict) and row.get("run_id") and row.get("cluster_id")}
    return {str(k): str(v) for k, v in data.items() if v is not None}


def _append_event_line(events_path: Path, event: dict[str, Any]) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as fh:
        fh.write(json.dumps(event, sort_keys=True) + "\n")


def run_monitor_once(
    *,
    control_path: Path,
    launch_plan_path: Path | None,
    state_path: Path,
    events_path: Path,
    condor_status_by_cluster: dict[str, dict[str, Any]] | None = None,
    better_analyze_by_cluster: dict[str, dict[str, Any]] | None = None,
    evidence_by_run_id: dict[str, dict[str, Any]] | None = None,
    cluster_id_by_run_id: dict[str, str] | None = None,
    monitor_only: bool = True,
    now: str | None = None,
    notify_file: Path | None = None,
) -> dict[str, Any]:
    now = now or utc_now_iso()
    control = load_json(control_path)
    launch_plan = load_json(launch_plan_path) if launch_plan_path else {}
    state = _seed_state(control, launch_plan, load_json(state_path), now)
    plan_rows = _runs_by_id(launch_plan)
    events_emitted: list[dict[str, Any]] = []
    state_events: dict[str, Any] = {}

    for run_id in _run_ids(control, launch_plan):
        previous = state.get("runs", {}).get(run_id)
        plan_row = plan_rows.get(run_id, {})
        cluster_id = plan_row.get("cluster_id") or (cluster_id_by_run_id or {}).get(run_id) or (previous or {}).get("cluster_id")
        condor_fields = None
        if cluster_id is not None:
            merged_condor_fields: dict[str, Any] = {}
            for key in ("submit_tier", "model_file", "module_name"):
                if plan_row.get(key) is not None:
                    evidence_key = "tier" if key == "submit_tier" else key
                    merged_condor_fields[evidence_key] = plan_row.get(key)
            if plan_row.get("batch_size") is not None:
                merged_condor_fields["batch_size"] = plan_row.get("batch_size")
            if condor_status_by_cluster:
                merged_condor_fields.update(condor_status_by_cluster.get(str(cluster_id)) or {})
            if better_analyze_by_cluster and str(cluster_id) in better_analyze_by_cluster:
                merged_condor_fields["better_analyze"] = better_analyze_by_cluster[str(cluster_id)]
            condor_fields = merged_condor_fields or None
        result = collect_and_classify_run(
            remote_root=state["remote_root"],
            campaign=state["campaign"],
            run_id=run_id,
            log_root=state["log_root"],
            condor_fields=condor_fields,
            evidence_override=(evidence_by_run_id or {}).get(run_id),
            plan_row=plan_row,
        )
        current = {
            "base_run_id": run_id,
            "current_run_id": run_id,
            "cluster_id": str(cluster_id) if cluster_id is not None else None,
            "state_key": result["state_key"],
            "classification": result["classification"],
            "plan": result["plan"],
            "last_evidence_summary": result["evidence_summary"],
            "repair_count": int(plan_row.get("repair_count", 0) or 0),
            "retry_count": int(plan_row.get("retry_count", 0) or 0),
            "total_attempts": int(plan_row.get("total_attempts", 1) or 1),
            "limits": plan_row.get("limits") if isinstance(plan_row.get("limits"), dict) else {"max_retries": 3, "max_repairs": 2, "max_total_attempts": 5},
            "updated_at": now,
            "notify_count": int((previous or {}).get("notify_count", 0)),
        }
        if should_emit_transition(previous, current):
            current["last_transition_at"] = now
            current["notify_count"] += 1
            event = {
                "time": now,
                "campaign": state["campaign"],
                "run_id": run_id,
                "previous_state_key": (previous or {}).get("state_key"),
                "state_key": current["state_key"],
                "classification": result["classification"].get("classification"),
                "action": result["plan"].get("action"),
                "monitor_only": monitor_only,
            }
            append_event(state_events, event)
            _append_event_line(events_path, event)
            events_emitted.append(event)
        else:
            current["last_transition_at"] = (previous or {}).get("last_transition_at")
        state["runs"][run_id] = current

    write_json(state_path, state)
    if notify_file and events_emitted:
        append_notifications(notify_file, events_emitted)
    return {"state": state, "events_emitted": events_emitted, "all_terminal": all_runs_terminal(state, monitor_only=monitor_only)}


def is_terminal_for_monitor(plan: dict[str, Any], *, monitor_only: bool) -> bool:
    action = str(plan.get("action") or "")
    if action in {ACTION_RECORD_RESULT, ACTION_AUTO_FAIL}:
        return True
    if monitor_only and action in {ACTION_RETRY, ACTION_REPAIR}:
        return True
    return False


def all_runs_terminal(state: dict[str, Any], *, monitor_only: bool) -> bool:
    runs = state.get("runs") or {}
    if not runs:
        return False
    return all(is_terminal_for_monitor(row.get("plan") or {}, monitor_only=monitor_only) for row in runs.values())


def append_notifications(notify_file: Path, events: list[dict[str, Any]]) -> None:
    notify_file.parent.mkdir(parents=True, exist_ok=True)
    with notify_file.open("a") as fh:
        for event in events:
            fh.write(f"- {event['time']} `{event['run_id']}` -> `{event['state_key']}` action=`{event['action']}`\n")


def run_ssh_command(command: str, *, ssh_host: str, ssh_control_path: str, runner: CommandRunner | None = None) -> subprocess.CompletedProcess[str]:
    runner = runner or (lambda args: subprocess.run(args, text=True, capture_output=True, check=False))
    return runner([
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=120",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        f"ControlPath={ssh_control_path}",
        ssh_host,
        command,
    ])


def parse_condor_q_af(stdout: str) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for line in stdout.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 3:
            continue
        cluster_id, proc_id, job_status = parts[:3]
        remote_host = parts[3] if len(parts) >= 4 else ""
        qdate: int | None = None
        hold_reason = ""
        if len(parts) >= 5:
            try:
                qdate = int(parts[4])
                hold_reason = parts[5] if len(parts) >= 6 else ""
            except ValueError:
                hold_reason = parts[4] if len(parts) == 5 else " ".join(parts[4:])
        row = {
            "cluster_id": cluster_id,
            "proc_id": proc_id,
            "condor_job_status": job_status,
            "remote_host": remote_host,
            "hold_reason": hold_reason,
        }
        if qdate is not None:
            row["qdate"] = qdate
        statuses[cluster_id] = row
    return statuses


def parse_better_analyze_summary(stdout: str) -> dict[str, int | None]:
    import re

    able_match = re.search(r"(\d+)\s+machines?\s+are\s+able\s+to\s+run\s+your\s+job", stdout, re.IGNORECASE)
    slots_match = re.search(r"(\d+)\s+slots?\s+match\s+your\s+job\s+requirements", stdout, re.IGNORECASE)
    return {
        "able_machines": int(able_match.group(1)) if able_match else None,
        "matched_slots": int(slots_match.group(1)) if slots_match else None,
    }


def poll_condor_statuses(
    cluster_ids: list[str],
    *,
    ssh_host: str,
    ssh_control_path: str,
    runner: CommandRunner | None = None,
    now_epoch: float | None = None,
) -> dict[str, dict[str, Any]]:
    if not cluster_ids:
        return {}
    ids = " ".join(str(x) for x in cluster_ids)
    cmd = f"condor_q {ids} -af ClusterId ProcId JobStatus RemoteHost QDate HoldReason"
    result = run_ssh_command(cmd, ssh_host=ssh_host, ssh_control_path=ssh_control_path, runner=runner)
    if result.returncode != 0:
        return {}
    statuses = parse_condor_q_af(result.stdout)
    now = time.time() if now_epoch is None else float(now_epoch)
    for row in statuses.values():
        if row.get("condor_job_status") in {"I", "1", 1} and row.get("qdate") is not None:
            try:
                row["idle_seconds"] = max(0, int(now - int(row["qdate"])))
            except (TypeError, ValueError):
                pass
    return statuses


def poll_better_analyze_summaries(
    cluster_ids: list[str],
    *,
    ssh_host: str,
    ssh_control_path: str,
    runner: CommandRunner | None = None,
) -> dict[str, dict[str, int | None]]:
    summaries: dict[str, dict[str, int | None]] = {}
    for cluster_id in cluster_ids:
        result = run_ssh_command(
            f"condor_q {cluster_id} -better-analyze",
            ssh_host=ssh_host,
            ssh_control_path=ssh_control_path,
            runner=runner,
        )
        if result.returncode == 0:
            summaries[str(cluster_id)] = parse_better_analyze_summary(result.stdout)
    return summaries


REMOTE_EVIDENCE_SCRIPT = r'''
import json, pathlib, re, sys
run_dir = pathlib.Path(sys.argv[1])
log_dir = pathlib.Path(sys.argv[2])
def read_tail(path, limit=40000):
    if not path.exists() or not path.is_file():
        return ""
    data = path.read_bytes()[-limit:]
    return data.decode("utf-8", errors="replace")
def read_json(path):
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
train_log = read_tail(run_dir / "train.log")
epochs = [int(m.group(1)) for m in re.finditer(r"\bEpoch\s+(\d+)\s*/\s*\d+", train_log)]
heartbeat = read_json(run_dir / "HEARTBEAT.json")
evidence = {
    "finished": (run_dir / "FINISHED").exists(),
    "failed": (run_dir / "FAILED").exists(),
    "metrics": read_json(run_dir / "metrics.json"),
    "heartbeat": heartbeat,
    "train_config": read_json(run_dir / "train_config.json"),
    "train_log": train_log,
    "condor_err": read_tail(log_dir / "condor.err"),
    "condor_out": read_tail(log_dir / "condor.out"),
    "condor_log": read_tail(log_dir / "condor.log"),
    "checkpoint_exists": (run_dir / "checkpoint.pt").exists(),
    "checkpoint_path": str(run_dir / "checkpoint.pt"),
}
if isinstance(heartbeat, dict):
    evidence["heartbeat_epoch"] = heartbeat.get("epoch")
    evidence["heartbeat_time"] = heartbeat.get("time")
if epochs:
    evidence["latest_train_epoch"] = max(epochs)
text = "\n".join(str(evidence.get(k, "")) for k in ("condor_err", "condor_out", "condor_log", "train_log"))
param_match = re.search(r"\bparams=(\d+)\b", text) or re.search(r"\bparams\s+(\d+)\s+limit\b", text)
if param_match:
    evidence["params"] = int(param_match.group(1))
if "FRONTEND_DYNAMIC_OK" in text:
    evidence["shape_ok"] = True
elif re.search(r"shape\s+(?:mismatch|fail|error)", text, re.IGNORECASE):
    evidence["shape_ok"] = False
print(json.dumps(evidence))
'''


BATCH_REMOTE_EVIDENCE_SCRIPT = r'''
import json, pathlib, re, sys
payload = json.loads(sys.argv[1])
def read_tail(path, limit=40000):
    if not path.exists() or not path.is_file():
        return ""
    data = path.read_bytes()[-limit:]
    return data.decode("utf-8", errors="replace")
def read_json(path):
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
result = {}
for item in payload:
    run_id = item["run_id"]
    run_dir = pathlib.Path(item["run_dir"])
    log_dir = pathlib.Path(item["log_dir"])
    train_log = read_tail(run_dir / "train.log")
    epochs = [int(m.group(1)) for m in re.finditer(r"\bEpoch\s+(\d+)\s*/\s*\d+", train_log)]
    heartbeat = read_json(run_dir / "HEARTBEAT.json")
    evidence = {
        "finished": (run_dir / "FINISHED").exists(),
        "failed": (run_dir / "FAILED").exists(),
        "metrics": read_json(run_dir / "metrics.json"),
        "heartbeat": heartbeat,
        "train_config": read_json(run_dir / "train_config.json"),
        "train_log": train_log,
        "condor_err": read_tail(log_dir / "condor.err"),
        "condor_out": read_tail(log_dir / "condor.out"),
        "condor_log": read_tail(log_dir / "condor.log"),
        "checkpoint_exists": (run_dir / "checkpoint.pt").exists(),
        "checkpoint_path": str(run_dir / "checkpoint.pt"),
    }
    if isinstance(heartbeat, dict):
        evidence["heartbeat_epoch"] = heartbeat.get("epoch")
        evidence["heartbeat_time"] = heartbeat.get("time")
    if epochs:
        evidence["latest_train_epoch"] = max(epochs)
    text = "\n".join(str(evidence.get(k, "")) for k in ("condor_err", "condor_out", "condor_log", "train_log"))
    param_match = re.search(r"\bparams=(\d+)\b", text) or re.search(r"\bparams\s+(\d+)\s+limit\b", text)
    if param_match:
        evidence["params"] = int(param_match.group(1))
    if "FRONTEND_DYNAMIC_OK" in text:
        evidence["shape_ok"] = True
    elif re.search(r"shape\s+(?:mismatch|fail|error)", text, re.IGNORECASE):
        evidence["shape_ok"] = False
    elif re.search(r"Device=.*GPU=.*params=\d+", text):
        evidence["shape_ok"] = True
    result[run_id] = evidence
print(json.dumps(result))
'''


def poll_remote_evidence(
    *,
    remote_root: str,
    campaign: str,
    run_ids: list[str],
    log_root: str,
    ssh_host: str,
    ssh_control_path: str,
    runner: CommandRunner | None = None,
) -> dict[str, dict[str, Any]]:
    payload = []
    for run_id in run_ids:
        run_dir, log_dir = build_run_paths(remote_root, campaign, run_id, log_root)
        payload.append({"run_id": run_id, "run_dir": str(run_dir), "log_dir": str(log_dir)})
    if not payload:
        return {}
    command = f"python3 -c {shlex.quote(BATCH_REMOTE_EVIDENCE_SCRIPT)} {shlex.quote(json.dumps(payload))}"
    result = run_ssh_command(command, ssh_host=ssh_host, ssh_control_path=ssh_control_path, runner=runner)
    if result.returncode != 0:
        return {}
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(run_id): evidence for run_id, evidence in parsed.items() if isinstance(evidence, dict)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Auto V5 monitor-only controller orchestrator; CRC access is opt-in via --live-crc")
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--launch-plan", type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--notify-file", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--monitor-only", action="store_true")
    parser.add_argument("--live-crc", action="store_true", help="Opt in to read-only CRC SSH polling for Condor status and run evidence")
    parser.add_argument("--ssh-host", default="lhu1@<HPC_FILE_LOGIN>")
    parser.add_argument("--ssh-control-path", default="<SSH_CONTROL_PATH>")
    parser.add_argument("--cluster-map", type=Path, help="Optional JSON mapping run_id to cluster_id, used when launch plan lacks cluster ids")
    parser.add_argument("--poll-interval-sec", type=float, default=600.0)
    parser.add_argument("--max-polls", type=int, default=1)
    parser.add_argument("--stop-when-terminal", action="store_true")
    args = parser.parse_args(argv)

    polls = 1 if args.once else max(1, args.max_polls)
    result: dict[str, Any] | None = None
    for idx in range(polls):
        condor_statuses = None
        better_summaries = None
        remote_evidence = None
        cluster_map = _cluster_ids_from_map(args.cluster_map)
        if args.live_crc:
            control = load_json(args.control)
            launch_plan = load_json(args.launch_plan) if args.launch_plan else {}
            run_ids = _run_ids(control, launch_plan)
            campaign = str(control.get("campaign") or launch_plan.get("campaign") or "")
            remote_root = str(control.get("remote_root") or launch_plan.get("remote_root") or "")
            log_root = str(control.get("log_root") or f"/users/lhu1/condor_v5_logs/{campaign}")
            plan_rows = _runs_by_id(launch_plan)
            cluster_ids = []
            for run_id in run_ids:
                cluster_id = plan_rows.get(run_id, {}).get("cluster_id") or cluster_map.get(run_id)
                if cluster_id:
                    cluster_ids.append(str(cluster_id))
            condor_statuses = poll_condor_statuses(cluster_ids, ssh_host=args.ssh_host, ssh_control_path=args.ssh_control_path)
            idle_cluster_ids = [cluster_id for cluster_id, row in condor_statuses.items() if row.get("condor_job_status") in {"I", "1", 1}]
            better_summaries = poll_better_analyze_summaries(idle_cluster_ids, ssh_host=args.ssh_host, ssh_control_path=args.ssh_control_path)
            remote_evidence = poll_remote_evidence(
                remote_root=remote_root,
                campaign=campaign,
                run_ids=run_ids,
                log_root=log_root,
                ssh_host=args.ssh_host,
                ssh_control_path=args.ssh_control_path,
            )
        result = run_monitor_once(
            control_path=args.control,
            launch_plan_path=args.launch_plan,
            state_path=args.state,
            events_path=args.events,
            condor_status_by_cluster=condor_statuses,
            better_analyze_by_cluster=better_summaries,
            evidence_by_run_id=remote_evidence,
            cluster_id_by_run_id=cluster_map,
            monitor_only=args.monitor_only,
            notify_file=args.notify_file,
        )
        if args.stop_when_terminal and result.get("all_terminal"):
            break
        if idx + 1 < polls:
            time.sleep(args.poll_interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
