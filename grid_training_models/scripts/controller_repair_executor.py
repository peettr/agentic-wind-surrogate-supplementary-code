#!/usr/bin/env python3
"""Stage, materialize, and optionally submit controller-planned Grid repair attempts.

This executor is separate from monitor-only orchestration. By default it writes a
repair control file and does not submit Condor jobs, call AI tools, SSH, or edit
train_config.json unless explicit opt-in flags are provided.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.launch_smoke_from_control import BOOKKEEPING_FIELDS, apply_generated_wrapper_source_of_truth, build_plan, submit_runs
from scripts.repair_code_stage import write_repair_stage

REPAIR_BOOKKEEPING_FIELDS = BOOKKEEPING_FIELDS | {
    "repair_of",
    "repair_reason",
    "repair_note",
    "repair_status",
    "repair_id",
    "repair_cluster",
    "repair_tier",
}


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


def _repair_budget_allows(entry: dict[str, Any]) -> bool:
    limits = entry.get("limits") if isinstance(entry.get("limits"), dict) else {}
    max_repairs = int(limits.get("max_repairs", entry.get("max_repairs", 2)))
    max_total = int(limits.get("max_total_attempts", entry.get("max_total_attempts", 5)))
    repair_count = _entry_repair_count(entry)
    total_attempts = _entry_total_attempts(entry)
    return repair_count < max_repairs and total_attempts < max_total


def _count_suffix(run_id: str, token: str) -> int:
    return len(re.findall(rf"_{re.escape(token)}\d+", run_id))


def _entry_repair_count(entry: dict[str, Any]) -> int:
    if entry.get("repair_count") is not None:
        return int(entry.get("repair_count") or 0)
    repairs = entry.get("repairs") if isinstance(entry.get("repairs"), list) else []
    if repairs:
        return len(repairs)
    source_run_id = str(entry.get("current_run_id") or entry.get("base_run_id") or "")
    return _count_suffix(source_run_id, "repair")


def _entry_retry_count(entry: dict[str, Any]) -> int:
    if entry.get("retry_count") is not None:
        return int(entry.get("retry_count") or 0)
    attempts = entry.get("attempts") if isinstance(entry.get("attempts"), list) else []
    if attempts:
        return sum(1 for row in attempts if isinstance(row, dict) and row.get("type") == "retry")
    source_run_id = str(entry.get("current_run_id") or entry.get("base_run_id") or "")
    return _count_suffix(source_run_id, "retry")


def _entry_total_attempts(entry: dict[str, Any]) -> int:
    if entry.get("total_attempts") is not None:
        return int(entry.get("total_attempts") or 1)
    attempts = entry.get("attempts") if isinstance(entry.get("attempts"), list) else []
    if attempts:
        return len(attempts)
    return 1 + _entry_repair_count(entry) + _entry_retry_count(entry)


def _text_blob(*parts: Any) -> str:
    chunks: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (dict, list)):
            chunks.append(json.dumps(part, sort_keys=True).lower())
        else:
            chunks.append(str(part).lower())
    return "\n".join(chunks)


def infer_deterministic_config_overrides(*, entry: dict[str, Any], launch_entry: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any] | None:
    """Return audited config-only repairs for known safe HIGH_VRAM patterns.

    These repairs intentionally avoid editing shared generated model files. They
    only write per-run ``arch_kwargs`` overrides into repair train_config.json,
    which lets the monitor/repair loop execute automatically without invoking
    live AI agents or modifying shared code used by sibling runs.
    """
    if isinstance(plan.get("config_overrides"), dict):
        return None
    classification = entry.get("classification") if isinstance(entry.get("classification"), dict) else {}
    evidence = entry.get("last_evidence_summary") if isinstance(entry.get("last_evidence_summary"), dict) else {}
    reason = str(plan.get("classification") or classification.get("classification") or "")
    action = str(plan.get("action") or classification.get("next_action") or "")
    text = _text_blob(reason, action, classification, evidence, launch_entry)
    batch_size = plan.get("batch_size") or launch_entry.get("batch_size") or evidence.get("batch_size")
    tier = str(plan.get("tier") or launch_entry.get("submit_tier") or evidence.get("tier") or "")
    model_file = str(launch_entry.get("model_file") or entry.get("model_file") or "")
    source_run_id = str(entry.get("current_run_id") or entry.get("base_run_id") or "")

    is_high_vram_repair = reason == "HIGH_VRAM" and action == "REPAIR"
    is_min_batch_oom = "oom" in text and ("batch=8" in text or "batch_size=8" in text or str(batch_size) == "8")
    is_high_tier = "h100" in tier or "a100" in tier or "l40s" in tier
    is_afno = "afno" in model_file.lower() or "afno" in source_run_id.lower() or "afno" in text
    if is_high_vram_repair and is_min_batch_oom and is_high_tier and is_afno:
        return {"arch_kwargs": {"width": 32, "depth": 3, "max_modes": 8}}
    is_attention_or_cbam = any(token in (model_file.lower() + " " + source_run_id.lower() + " " + text) for token in ("attention_gate_unet", "cbam_unet"))
    if reason == "PARAM_TOO_LARGE" and action == "REPAIR" and is_attention_or_cbam:
        return {"arch_kwargs": {"n_c": 16, "depth": 6, "training": {"data_augment": False}}}
    return None


def planned_repair_runs(
    *,
    control: dict[str, Any],
    launch_plan: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    campaign = str(control.get("campaign") or launch_plan.get("campaign") or state.get("campaign") or "")
    if not campaign:
        raise SystemExit("control, launch plan, or state must define campaign")
    by_run = run_lookup(launch_plan)
    state_runs = state.get("runs", {})
    if not isinstance(state_runs, dict):
        raise SystemExit("controller state runs must be an object")

    planned: list[dict[str, Any]] = []
    used_repair_ids: set[str] = set()
    for state_run_id, entry in state_runs.items():
        if not isinstance(entry, dict):
            continue
        plan = entry.get("plan", {})
        if not isinstance(plan, dict) or plan.get("action") != "REPAIR":
            continue
        if not _repair_budget_allows(entry):
            continue
        source_run_id = str(entry.get("current_run_id") or state_run_id)
        launch_entry = by_run.get(source_run_id, {})
        new_run_id = str(plan.get("new_run_id") or "")
        if not new_run_id:
            raise SystemExit(f"{source_run_id}: repair plan missing new_run_id")
        desired_repair_count = _entry_repair_count(entry) + 1
        repair_id = str(plan.get("repair_id") or f"repair_{desired_repair_count:03d}")
        if desired_repair_count > 1:
            repair_id = f"repair_{desired_repair_count:03d}"
        original_repair_id = repair_id
        if repair_id in used_repair_ids:
            slug_source = source_run_id or new_run_id
            slug = re.sub(r"[^A-Za-z0-9_]+", "_", slug_source).strip("_") or f"{len(planned) + 1:03d}"
            repair_id = f"{original_repair_id}_{slug}"
            suffix = 2
            while repair_id in used_repair_ids:
                repair_id = f"{original_repair_id}_{slug}_{suffix}"
                suffix += 1
        used_repair_ids.add(repair_id)
        repair_run = {
            "source_campaign": campaign,
            "source_run_id": source_run_id,
            "run_id": new_run_id,
            "model_file": launch_entry.get("model_file") or entry.get("model_file"),
            "module_name": launch_entry.get("module_name") or entry.get("module_name"),
            "submit_tier": plan.get("tier") or launch_entry.get("submit_tier") or entry.get("last_evidence_summary", {}).get("tier"),
            "batch_size": plan.get("batch_size") or launch_entry.get("batch_size") or entry.get("last_evidence_summary", {}).get("batch_size"),
            "reason": plan.get("classification") or entry.get("classification", {}).get("classification") or "REPAIR",
            "repair_id": repair_id,
            "repair_count": _entry_repair_count(entry) + 1,
            "retry_count": _entry_retry_count(entry),
            "total_attempts": _entry_total_attempts(entry) + 1,
            "limits": {
                "max_retries": int(entry.get("max_retries", 3) or 3),
                "max_repairs": int(entry.get("max_repairs", 2) or 2),
                "max_total_attempts": int(entry.get("max_total_attempts", 5) or 5),
            },
        }
        if launch_entry.get("allow_param_cap_relaxation") or plan.get("allow_param_cap_relaxation"):
            repair_run["allow_param_cap_relaxation"] = True
        if isinstance(plan.get("config_overrides"), dict):
            repair_run["config_overrides"] = plan["config_overrides"]
        else:
            inferred_overrides = infer_deterministic_config_overrides(entry=entry, launch_entry=launch_entry, plan=plan)
            if inferred_overrides:
                repair_run["config_overrides"] = inferred_overrides
                repair_run["repair_note"] = f"deterministic_config_repair: {entry.get('classification', {}).get('classification') or plan.get('classification') or 'classified'} safe downsizing"
        planned.append(repair_run)
    return planned


def repair_context_for_run(
    *,
    control: dict[str, Any],
    state: dict[str, Any],
    repair_run: dict[str, Any],
    local_root: Path | None = None,
) -> dict[str, Any]:
    source_run_id = str(repair_run["source_run_id"])
    entry = state.get("runs", {}).get(source_run_id, {})
    if not isinstance(entry, dict):
        entry = {}
    evidence = entry.get("last_evidence_summary") if isinstance(entry.get("last_evidence_summary"), dict) else {}
    classification = entry.get("classification") if isinstance(entry.get("classification"), dict) else {}
    plan = entry.get("plan") if isinstance(entry.get("plan"), dict) else {}
    source_control_row = None
    for row in control.get("runs", []):
        if isinstance(row, dict) and row.get("run_id") == source_run_id:
            source_control_row = row
            break
    file_snapshots: list[dict[str, str]] = []
    if local_root is not None:
        candidates = [
            (str(repair_run.get("model_file") or ""), "python"),
            (f"campaigns/{control.get('campaign')}/runs/{source_run_id}/train_config.json", "json"),
            (f"campaigns/{control.get('campaign')}/runs/{repair_run['run_id']}/train_config.json", "json"),
        ]
        seen: set[str] = set()
        for rel, language in candidates:
            if not rel or rel in seen:
                continue
            seen.add(rel)
            path = local_root / rel
            if path.exists() and path.is_file():
                text = path.read_text(errors="replace")
                file_snapshots.append({"path": rel, "language": language, "content": text[:60000]})
    return {
        "base_run_id": entry.get("base_run_id") or source_run_id,
        "run_id": repair_run["run_id"],
        "campaign": control.get("campaign") or state.get("campaign"),
        "classification": repair_run.get("reason"),
        "model_file": repair_run.get("model_file"),
        "train_config": f"campaigns/{control.get('campaign')}/runs/{repair_run['run_id']}/train_config.json",
        "retry_count": entry.get("retry_count", 0),
        "repair_count": entry.get("repair_count", 0),
        "total_attempts": len(entry.get("attempts", [])) + 1 if isinstance(entry.get("attempts"), list) else 1,
        "limits": {"max_retries": 3, "max_repairs": 2, "max_total_attempts": 5},
        "forbidden_changes": [
            "batch_size < 8",
            "epochs != 20 for smoke20",
            "bookkeeping fields in train_config.json",
            "modify only the generated standalone model file by default; shared code requires explicit human approval",
        ],
        "allowed_change_paths": [repair_run.get("model_file")],
        "forbidden_change_paths": ["shared/train.py", "shared/models"],
        "evidence": evidence,
        "classification_details": classification,
        "planned_action": plan,
        "attempts": entry.get("attempts", []),
        "logs": entry.get("logs", {}),
        "controller_state_entry": entry,
        "repair_run_entry": repair_run,
        "source_control_row": source_control_row,
        "file_snapshots": file_snapshots,
    }


def stage_repair_runs(*, control: dict[str, Any], state: dict[str, Any], runs: list[dict[str, Any]], output_root: Path, local_root: Path | None = None) -> None:
    for run in runs:
        context = repair_context_for_run(control=control, state=state, repair_run=run, local_root=local_root)
        write_repair_stage(context, repair_id=str(run["repair_id"]), output_root=output_root, execute=False)


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"train_config root must be object: {path}")
    return data


def _deep_merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def materialize_repair_run(*, local_root: Path, campaign: str, remote_root: str, run: dict[str, Any]) -> None:
    source_run_id = str(run["source_run_id"])
    new_run_id = str(run["run_id"])
    source_config = local_root / "campaigns" / campaign / "runs" / source_run_id / "train_config.json"
    output_dir = local_root / "campaigns" / campaign / "runs" / new_run_id
    cfg = _load_config(source_config)
    cfg = {k: v for k, v in cfg.items() if k not in REPAIR_BOOKKEEPING_FIELDS}
    overrides = run.get("config_overrides")
    if isinstance(overrides, dict):
        forbidden = REPAIR_BOOKKEEPING_FIELDS & set(overrides)
        if forbidden:
            raise SystemExit(f"config_overrides contains bookkeeping fields for {new_run_id}: {sorted(forbidden)}")
        cfg = _deep_merge(cfg, overrides)
    cfg["experiment_id"] = new_run_id
    cfg["results_dir"] = f"{remote_root.rstrip('/')}/campaigns/{campaign}/runs/{new_run_id}"
    apply_generated_wrapper_source_of_truth(cfg, remote_root=remote_root, row=run)
    if run.get("batch_size") is not None:
        batch_size = int(run["batch_size"])
        if batch_size < 8:
            raise SystemExit(f"batch_size below 8 is not allowed for {new_run_id}")
        cfg["batch_size"] = batch_size

    # Validate with the same schema gate as launcher materialization.
    from shared.configs.schema import TrainConfig
    TrainConfig.model_validate(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    note = {
        "source_config": str(source_config),
        "source_run_id": source_run_id,
        "new_run_id": new_run_id,
        "classification": run.get("reason"),
        "reason": run.get("reason"),
        "repair_id": run.get("repair_id"),
        "tier": run.get("submit_tier"),
        "batch_size": cfg.get("batch_size"),
        "metadata_rule": "repair bookkeeping is stored outside train_config.json",
    }
    (output_dir / "REPAIR_NOTE.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in note.items() if v is not None) + "\n"
    )


def _paths_from_repair_context(context_path: Path) -> list[str]:
    if not context_path.exists():
        return []
    paths: list[str] = []
    in_allowed = False
    for line in context_path.read_text(errors="replace").splitlines():
        if line.startswith("model_file:") or line.startswith("train_config:"):
            _, value = line.split(":", 1)
            value = value.strip()
            if value:
                paths.append(value)
            continue
        if line.strip() == "## Allowed change paths":
            in_allowed = True
            continue
        if in_allowed and line.startswith("## "):
            in_allowed = False
        if in_allowed and line.strip().startswith("- "):
            value = line.strip()[2:].strip()
            if value:
                paths.append(value)
    deduped: list[str] = []
    for path in paths:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _append_step_context_update(*, context_path: Path | None, cwd: Path, result: dict[str, Any], max_chars: int = 12000) -> None:
    if context_path is None or not context_path.exists():
        return
    paths = _paths_from_repair_context(context_path)
    lines = [
        "",
        "## AI repair execution history",
        f"### Completed step: {result.get('agent')} / {result.get('role')}",
        "```json",
        json.dumps({
            "repair_id": result.get("repair_id"),
            "agent": result.get("agent"),
            "role": result.get("role"),
            "returncode": result.get("returncode"),
            "stdout_tail": result.get("stdout_tail"),
            "stderr_tail": result.get("stderr_tail"),
        }, indent=2),
        "```",
    ]
    if paths:
        lines.append("### Refreshed relevant file snapshots after this step")
    for rel in paths:
        file_path = (cwd / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        try:
            content = file_path.read_text(errors="replace")
        except OSError as exc:
            content = f"<unavailable: {exc}>"
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n... [truncated to {max_chars} chars]"
        lines.extend([f"#### File: {rel}", "```text", content, "```"])
    with context_path.open("a") as fh:
        fh.write("\n".join(lines) + "\n")


def execute_staged_repair_steps(*, repair_output_root: Path, runs: list[dict[str, Any]], cwd: Path, dry_run: bool, timeout_sec: int = 900) -> list[dict[str, Any]]:
    """Execute staged AI repair commands only when explicitly requested.

    This is intentionally small and auditable. It reads the commands produced by
    repair_code_stage.py and runs them sequentially. Tests and dry runs can call
    the same path without side effects by passing dry_run=True. AI repair commands
    are bounded by timeout_sec so a hung agent cannot stall the controller clock.
    """
    results: list[dict[str, Any]] = []
    for run in runs:
        commands_path = repair_output_root / str(run["repair_id"]) / "commands.json"
        plan = load_json(commands_path)
        context_path = Path(plan["context_path"]) if plan.get("context_path") else None
        for step in plan.get("steps", []):
            command = step.get("command")
            if not isinstance(command, list) or not command:
                raise SystemExit(f"invalid repair command in {commands_path}: {step}")
            result = {"repair_id": run["repair_id"], "agent": step.get("agent"), "role": step.get("role"), "command": command}
            if dry_run:
                result["dry_run"] = True
            else:
                try:
                    proc = subprocess.Popen(command, cwd=cwd, text=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                    stdout, stderr = proc.communicate(timeout=timeout_sec)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        stdout, stderr = proc.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        stdout, stderr = proc.communicate()
                    result.update({
                        "returncode": "TIMEOUT",
                        "timeout_sec": timeout_sec,
                        "stdout_tail": (stdout or "")[-4000:],
                        "stderr_tail": (stderr or "")[-4000:],
                    })
                    results.append(result)
                    raise SystemExit(json.dumps({"failed_repair_step": result}, indent=2))
                result.update({"returncode": proc.returncode, "stdout_tail": stdout[-4000:], "stderr_tail": stderr[-4000:]})
                _append_step_context_update(context_path=context_path, cwd=cwd, result=result)
                if proc.returncode != 0:
                    results.append(result)
                    raise SystemExit(json.dumps({"failed_repair_step": result}, indent=2))
            results.append(result)
    return results


def _extract_cluster_id(text: str) -> str | None:
    match = re.search(r"submitted to cluster\s+(\d+)", text or "", re.I)
    return match.group(1) if match else None


def attach_submit_results_to_plan(plan: dict[str, Any]) -> dict[str, str]:
    """Parse Condor cluster IDs from submit stdout and attach them to the plan.

    ``submit_runs`` keeps stdout/stderr tails in ``submit_results``. Repair
    monitoring needs a run_id -> cluster_id map, so this helper extracts cluster
    IDs from the standard Condor text and mirrors them onto both result rows and
    plan rows.
    """
    results = plan.get("submit_results") if isinstance(plan.get("submit_results"), list) else []
    cluster_by_run: dict[str, str] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        run_id = result.get("run_id")
        cluster_id = result.get("cluster_id") or _extract_cluster_id(str(result.get("stdout_tail") or ""))
        if isinstance(run_id, str) and cluster_id:
            result["cluster_id"] = str(cluster_id)
            cluster_by_run[run_id] = str(cluster_id)
    for row in plan.get("runs", []):
        if isinstance(row, dict) and isinstance(row.get("run_id"), str) and row["run_id"] in cluster_by_run:
            row["cluster_id"] = cluster_by_run[row["run_id"]]
    return cluster_by_run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--launch-plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, default=Path("."))
    parser.add_argument("--repair-control-output", type=Path, required=True)
    parser.add_argument("--repair-output-root", type=Path, default=Path("reports/repairs"))
    parser.add_argument("--remote-root", default=None, help="Override the remote project root used in repair train_config results_dir and submit plan.")
    parser.add_argument("--stage-repair", action="store_true", help="Write repair context.md and commands.json for planned repair runs.")
    parser.add_argument("--execute-repair", action="store_true", help="Explicit opt-in to execute staged Claude/Codex/Claude repair commands.")
    parser.add_argument("--materialize", action="store_true", help="Write sanitized repair train_config.json files and REPAIR_NOTE sidecars.")
    parser.add_argument("--submit-repair", action="store_true", help="Explicit opt-in to submit materialized repair runs through crc_codegen_smoke_one.sh.")
    parser.add_argument("--submitted-plan-output", type=Path, default=None)
    parser.add_argument("--cluster-map-output", type=Path, default=None, help="Optional JSON output mapping repair run_id to submitted Condor cluster_id.")
    parser.add_argument("--execution-log-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.submit_repair and not args.materialize:
        raise SystemExit("--submit-repair requires --materialize so repair configs are validated before submission")
    if args.execute_repair and not args.stage_repair:
        raise SystemExit("--execute-repair requires --stage-repair so repair commands are explicit and inspectable")

    control = load_json(args.control)
    launch_plan = load_json(args.launch_plan)
    state = load_json(args.state)
    campaign = str(control.get("campaign") or launch_plan.get("campaign") or state.get("campaign"))
    remote_root = str(args.remote_root or control.get("remote_root") or launch_plan.get("remote_root") or state.get("remote_root") or "")
    if not campaign or not remote_root:
        raise SystemExit("campaign and remote_root are required from control, launch plan, or state")

    runs = planned_repair_runs(control=control, launch_plan=launch_plan, state=state)
    repair_control = {
        "campaign": campaign,
        "remote_root": remote_root,
        "stage": control.get("stage") or launch_plan.get("stage") or state.get("stage") or "smoke20",
        "runs": runs,
    }
    write_json(args.repair_control_output, repair_control)

    if args.stage_repair:
        stage_repair_runs(control=control, state=state, runs=runs, output_root=args.repair_output_root, local_root=args.local_root)

    if args.execute_repair:
        execution_log = {"runs": execute_staged_repair_steps(repair_output_root=args.repair_output_root, runs=runs, cwd=args.local_root, dry_run=args.dry_run)}
        if args.execution_log_output:
            write_json(args.execution_log_output, execution_log)

    if args.materialize:
        for run in runs:
            materialize_repair_run(local_root=args.local_root, campaign=campaign, remote_root=remote_root, run=run)

    submitted_plan: dict[str, Any] | None = None
    if args.submit_repair:
        submitted_plan = build_plan(args.local_root, repair_control, materialize=False)
        if not args.dry_run:
            submitted_plan["submit_results"] = submit_runs(args.local_root, submitted_plan)
            cluster_map = attach_submit_results_to_plan(submitted_plan)
            if args.cluster_map_output:
                write_json(args.cluster_map_output, cluster_map)
        if args.submitted_plan_output:
            write_json(args.submitted_plan_output, submitted_plan)

    print(f"Planned {len(runs)} repair run(s)")
    print(f"Wrote repair control {args.repair_control_output}")
    if args.stage_repair:
        print(f"Staged repair contexts under {args.repair_output_root}")
    if args.execute_repair:
        if args.dry_run:
            print("Execute-repair dry-run only; no AI commands were executed")
        else:
            print("Executed staged AI repair commands")
    if args.submit_repair:
        if args.dry_run:
            print("Submit-repair dry-run only; no Condor submit was executed")
        else:
            submitted_count = len(submitted_plan.get("submit_results", [])) if submitted_plan else 0
            print(f"Submitted {submitted_count} repair run(s)")
    elif args.materialize:
        print("Materialized sanitized repair configs")
    else:
        print("Dry-run only; no train_config.json files were written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



