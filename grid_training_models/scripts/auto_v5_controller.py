#!/usr/bin/env python3
"""Deterministic helpers for Auto V5 retry/repair controller state.

This module owns policy constants and manifest accounting. It deliberately does
not call AI tools or Condor; higher-level scripts use these pure helpers before
performing side effects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MAX_RETRIES = 3
MAX_REPAIRS = 2
MAX_TOTAL_ATTEMPTS = 5

AUTO_FAIL_MAX_RETRIES = "AUTO_FAIL_MAX_RETRIES"
AUTO_FAIL_MAX_REPAIRS = "AUTO_FAIL_MAX_REPAIRS"
AUTO_FAIL_MAX_TOTAL_ATTEMPTS = "AUTO_FAIL_MAX_TOTAL_ATTEMPTS"

ACTION_RETRY = "RETRY"
ACTION_REPAIR = "REPAIR"
ACTION_AUTO_FAIL = "AUTO_FAIL"
ACTION_WAIT = "WAIT"
ACTION_RECORD_RESULT = "RECORD_RESULT"
ACTION_COLLECT_MORE_EVIDENCE = "COLLECT_MORE_EVIDENCE"


def decide_limit_status(*, total_attempts: int, retry_count: int, repair_count: int, next_action: str) -> str | None:
    """Return an AUTO_FAIL status if the requested next action exceeds limits."""
    action = next_action.upper()
    if total_attempts >= MAX_TOTAL_ATTEMPTS:
        return AUTO_FAIL_MAX_TOTAL_ATTEMPTS
    if action == "RETRY" and retry_count >= MAX_RETRIES:
        return AUTO_FAIL_MAX_RETRIES
    if action == "REPAIR" and repair_count >= MAX_REPAIRS:
        return AUTO_FAIL_MAX_REPAIRS
    return None


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": []}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"manifest root must be object: {path}")
    data.setdefault("runs", [])
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def get_base_entry(manifest: dict[str, Any], base_run_id: str) -> dict[str, Any]:
    runs = manifest.setdefault("runs", [])
    for entry in runs:
        if entry.get("base_run_id") == base_run_id:
            entry.setdefault("attempts", [])
            entry.setdefault("repairs", [])
            _refresh_counts(entry)
            return entry
    entry = {
        "base_run_id": base_run_id,
        "max_retries": MAX_RETRIES,
        "max_repairs": MAX_REPAIRS,
        "max_total_attempts": MAX_TOTAL_ATTEMPTS,
        "retry_count": 0,
        "repair_count": 0,
        "total_attempts": 0,
        "status": "PENDING",
        "attempts": [],
        "repairs": [],
    }
    runs.append(entry)
    return entry


def count_attempts(entry: dict[str, Any]) -> tuple[int, int, int]:
    attempts = entry.get("attempts", []) or []
    total = len(attempts)
    retry_count = sum(1 for row in attempts if row.get("type") == "retry")
    repair_count = len(entry.get("repairs", []) or [])
    return total, retry_count, repair_count


def _refresh_counts(entry: dict[str, Any]) -> None:
    total, retry_count, repair_count = count_attempts(entry)
    entry["total_attempts"] = total
    entry["retry_count"] = retry_count
    entry["repair_count"] = repair_count
    entry.setdefault("max_retries", MAX_RETRIES)
    entry.setdefault("max_repairs", MAX_REPAIRS)
    entry.setdefault("max_total_attempts", MAX_TOTAL_ATTEMPTS)


def append_attempt(entry: dict[str, Any], attempt: dict[str, Any]) -> None:
    entry.setdefault("attempts", []).append(dict(attempt))
    _refresh_counts(entry)


def append_repair(entry: dict[str, Any], repair: dict[str, Any]) -> None:
    entry.setdefault("repairs", []).append(dict(repair))
    _refresh_counts(entry)


def _base_run_id(entry: dict[str, Any]) -> str:
    return str(entry["base_run_id"])


def next_retry_run_id(entry: dict[str, Any]) -> str:
    _, retry_count, _ = count_attempts(entry)
    return f"{_base_run_id(entry)}_retry{retry_count + 1}"


def next_repair_run_id(entry: dict[str, Any]) -> str:
    _, _, repair_count = count_attempts(entry)
    return f"{_base_run_id(entry)}_repair{repair_count + 1}"


def next_repair_id(entry: dict[str, Any]) -> str:
    _, _, repair_count = count_attempts(entry)
    return f"repair_{repair_count + 1:03d}"


def run_state_key(classification: dict[str, Any], plan: dict[str, Any]) -> str:
    """Return the stable state key used for transition detection."""
    cls = str(classification.get("classification") or "UNKNOWN")
    action = str(plan.get("action") or classification.get("next_action") or "UNKNOWN")
    return f"{cls}:{action}"


def should_emit_transition(previous: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    """Return True when the monitor should emit an event for this state."""
    if previous is None:
        return True
    return previous.get("state_key") != current.get("state_key")


def append_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    """Append a copy of an event to in-memory controller state."""
    state.setdefault("events", []).append(dict(event))


def plan_next_step(entry: dict[str, Any], classification: dict[str, Any]) -> dict[str, Any]:
    """Plan the next controller action without side effects."""
    _refresh_counts(entry)
    next_action = str(classification.get("next_action", "")).upper()
    limit_status = decide_limit_status(
        total_attempts=int(entry.get("total_attempts", 0)),
        retry_count=int(entry.get("retry_count", 0)),
        repair_count=int(entry.get("repair_count", 0)),
        next_action=next_action,
    )
    if limit_status:
        return {"action": ACTION_AUTO_FAIL, "status": limit_status, "classification": classification.get("classification")}
    if next_action == ACTION_RETRY:
        plan = {
            "action": ACTION_RETRY,
            "new_run_id": next_retry_run_id(entry),
            "tier": classification.get("recommended_tier"),
            "batch_size": classification.get("recommended_batch_size"),
            "classification": classification.get("classification"),
        }
        for key in (
            "resume_from_checkpoint",
            "checkpoint_epoch",
            "checkpoint_path",
            "last_epoch",
            "heartbeat_epoch",
            "heartbeat_age_sec",
            "walltime_sec",
            "condor_event",
        ):
            if key in classification:
                plan[key] = classification[key]
        if classification.get("classification") in {"EVICTED_WITH_CHECKPOINT", "TIMEOUT_OR_EVICTION_RESUMABLE"}:
            plan.setdefault("resume_from_checkpoint", True)
        return plan
    if next_action == ACTION_REPAIR:
        return {
            "action": ACTION_REPAIR,
            "new_run_id": next_repair_run_id(entry),
            "repair_id": next_repair_id(entry),
            "classification": classification.get("classification"),
        }
    return {"action": next_action, "classification": classification.get("classification")}
