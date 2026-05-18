"""Attempt manifest for Hybrid/V5 controller logic.

The manifest is per campaign and per semantic base run. It is the source of
truth for attempts, retries, repairs, limits, and terminal status. The runner
owns round-level orchestration; this module owns per-run accounting.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

MAX_RETRIES = 3
MAX_REPAIRS = 2
MAX_TOTAL_ATTEMPTS = 5

TERMINAL_STATUSES = {
    "PASS",
    "AUTO_FAIL_MAX_RETRIES",
    "AUTO_FAIL_MAX_REPAIRS",
    "AUTO_FAIL_MAX_TOTAL_ATTEMPTS",
    "AUTO_FAIL_UNKNOWN",
    "AUTO_FAIL_UNREPAIRABLE",
    "AUTO_FAIL_STALE_CONFIG",
    "AUTO_FAIL_RESOURCE_GUARD",
    "DUPLICATE_PASS",
}


def manifest_path(campaign_dir: Path) -> Path:
    return campaign_dir / "attempt_manifest.json"


def base_run_id(run_id: str) -> str:
    """Return semantic experiment id shared by initial/retry/repair attempts."""
    s = re.sub(r"^r\d+_(smoke|full)_", "", run_id)
    s = re.sub(r"_(retry|repair)\d+$", "", s)
    return s


def load_manifest(campaign_dir: Path) -> dict[str, Any]:
    p = manifest_path(campaign_dir)
    if not p.exists():
        return {"version": 1, "runs": {}}
    return json.loads(p.read_text(encoding="utf-8-sig"))


def save_manifest(campaign_dir: Path, manifest: dict[str, Any]) -> None:
    p = manifest_path(campaign_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    import os
    os.replace(tmp, p)


def ensure_run(manifest: dict[str, Any], run_id: str, run_type: str, config: dict | None = None) -> dict[str, Any]:
    base_id = base_run_id(run_id)
    runs = manifest.setdefault("runs", {})
    if base_id not in runs:
        runs[base_id] = {
            "base_run_id": base_id,
            "run_type": run_type,
            "max_retries": MAX_RETRIES,
            "max_repairs": MAX_REPAIRS,
            "max_total_attempts": MAX_TOTAL_ATTEMPTS,
            "retry_count": 0,
            "repair_count": 0,
            "total_attempts": 0,
            "status": "ACTIVE",
            "config": config or {},
            "attempts": [],
            "repairs": [],
        }
    return runs[base_id]


def record_attempt(
    manifest: dict[str, Any],
    run_id: str,
    run_type: str,
    attempt_type: str,
    status: str,
    classification: str,
    action: str,
    config: dict | None = None,
    cluster_id: str | None = None,
    evidence: list[str] | None = None,
    metrics_path: str | None = None,
) -> dict[str, Any]:
    entry = ensure_run(manifest, run_id, run_type, config)
    attempts = entry.setdefault("attempts", [])
    existing = next((a for a in attempts if a.get("run_id") == run_id), None)
    payload = {
        "attempt": existing.get("attempt") if existing else len(attempts),
        "run_id": run_id,
        "type": attempt_type,
        "status": status,
        "cluster_id": cluster_id,
        "classification": classification,
        "action": action,
        "evidence": evidence or [],
        "metrics_path": metrics_path,
    }
    if existing:
        existing.update(payload)
    else:
        attempts.append(payload)
        entry["total_attempts"] = len(attempts)
        if attempt_type == "retry":
            entry["retry_count"] = int(entry.get("retry_count", 0)) + 1
        elif attempt_type == "repair_rerun":
            entry["repair_count"] = int(entry.get("repair_count", 0)) + 1
    # PASS is the strongest terminal state â€” never downgrade to AUTO_FAIL.
    # A completed run with metrics is a success regardless of attempt budget.
    if classification == "PASS":
        # If this config already passed, mark as duplicate pass to avoid
        # penalizing the budget for repeated successful proposals across rounds.
        if entry.get("status") == "PASS" and not existing:
            payload["classification"] = "DUPLICATE_PASS"
            payload["action"] = "DUPLICATE_PASS"
        entry["status"] = "PASS"
        entry["metrics_path"] = metrics_path or entry.get("metrics_path")
    elif action.startswith("AUTO_FAIL") and entry.get("status") != "PASS":
        entry["status"] = action
        entry["terminal_reason"] = classification
    return entry


def check_limit(entry: dict[str, Any], next_action: str, current_attempt_type: str | None = None, *, current_classification: str | None = None) -> str | None:
    """Return AUTO_FAIL_* when the just-observed attempt exhausts a limit.

    Controller calls this before record_attempt() updates counters for the
    current run_id. Account for the current failed retry/repair attempt here;
    otherwise retryN can be submitted one extra time because the manifest only
    contains previously collected attempts.

    IMPORTANT: A PASS (completed with metrics) never triggers limit checks.
    Attempt budgets exist to prevent wasting resources on repeated failures,
    not to penalize successful completions that happen to reuse a config
    across multiple rounds.
    """
    # PASS is terminal-success; never override with a limit failure.
    if current_classification == "PASS" or next_action == "PASS":
        return None
    # If the entry already achieved PASS, do not downgrade to AUTO_FAIL.
    if entry.get("status") == "PASS":
        return None

    current_counts = current_attempt_type in {"initial", "retry", "repair_rerun"}
    effective_total = int(entry.get("total_attempts", 0)) + (1 if current_counts else 0)
    effective_retries = int(entry.get("retry_count", 0)) + (1 if current_attempt_type == "retry" else 0)
    effective_repairs = int(entry.get("repair_count", 0)) + (1 if current_attempt_type == "repair_rerun" else 0)

    if effective_total >= int(entry.get("max_total_attempts", MAX_TOTAL_ATTEMPTS)):
        return "AUTO_FAIL_MAX_TOTAL_ATTEMPTS"
    if next_action == "RETRY" and effective_retries >= int(entry.get("max_retries", MAX_RETRIES)):
        return "AUTO_FAIL_MAX_RETRIES"
    if next_action == "REPAIR" and effective_repairs >= int(entry.get("max_repairs", MAX_REPAIRS)):
        return "AUTO_FAIL_MAX_REPAIRS"
    return None


def mark_stale_active_runs(
    manifest: dict[str, Any],
    *,
    active_base_ids: set[str],
    round_prefix: str,
    reason: str = "not present in current proposal/result set",
) -> int:
    """Mark current-round ACTIVE entries as terminal when they were dropped.

    This keeps manifest summaries from showing stale ACTIVE rows after a bad
    proposal has been removed or renamed during manual/automatic recovery. Only
    entries with attempts from the current round are touched, and entries still
    represented by the current proposal/result semantic ids are preserved.
    """
    changed = 0
    for base_id, entry in manifest.get("runs", {}).items():
        if entry.get("status") != "ACTIVE":
            continue
        attempts = entry.get("attempts", []) or []
        touched_this_round = any(str(a.get("run_id", "")).startswith(round_prefix) for a in attempts)
        if touched_this_round and base_id not in active_base_ids:
            entry["status"] = "AUTO_FAIL_STALE_CONFIG"
            entry["terminal_reason"] = reason
            changed += 1
    return changed


def manifest_summary(manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in manifest.get("runs", {}).values():
        st = r.get("status", "UNKNOWN")
        counts[st] = counts.get(st, 0) + 1
    return counts




