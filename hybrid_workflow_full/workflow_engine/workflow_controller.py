"""Deterministic Hybrid/V5 controller.

Runs after collect and before reviewer. It updates per-run attempt manifest,
classifies collected evidence, checks retry/repair/total limits, and emits a
single controller_decision.json for the runner.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "workflow_engine"))

from workflow_common import load_state, round_artifact_dir, now_iso, experiment_id
from attempt_manifest import load_manifest, save_manifest, record_attempt, ensure_run, check_limit, manifest_summary, base_run_id, mark_stale_active_runs
from failure_classifier import classify_result


def _artifact_results(art_dir: Path, tag: str) -> list[dict]:
    p = art_dir / ("smoke_results.json" if tag == "smoke" else "full_results.json")
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _attempt_rank(run_id: str) -> int:
    import re
    m = re.search(r"_(?:retry|repair)(\d+)$", run_id)
    return int(m.group(1)) if m else 0


def _latest_semantic_results(results: list[dict]) -> list[dict]:
    """Collapse initial/retry/repair attempts to latest result per base run."""
    latest: dict[str, dict] = {}
    for r in results:
        rid = r.get("experiment_id") or r.get("exp_id") or "unknown"
        bid = base_run_id(rid)
        prev = latest.get(bid)
        if prev is None:
            latest[bid] = r
            continue
        prev_id = prev.get("experiment_id") or prev.get("exp_id") or "unknown"
        if _attempt_rank(rid) >= _attempt_rank(prev_id):
            latest[bid] = r
    return list(latest.values())


def _attempt_type(run_id: str, state: dict) -> str:
    if state.get("fix_mode") or "repair" in run_id:
        return "repair_rerun"
    if "retry" in run_id:
        return "retry"
    return "initial"


def decide(campaign_dir: Path) -> dict:
    state = load_state(campaign_dir)
    tag = state.get("last_collect_tag") or state.get("submit_tag") or "smoke"
    round_num = state.get("round_num", 0)
    art_dir = round_artifact_dir(campaign_dir, round_num)
    if tag == "smoke" and state.get("smoke_results"):
        # Smoke may collect initial attempts plus retry/repair attempts across
        # multiple ticks. Round-level decisions need the latest semantic result
        # for each proposed run, not only the most recent retry artifact.
        results = state.get("smoke_results", [])
    else:
        results = _artifact_results(art_dir, tag)
    results = _latest_semantic_results(results)

    manifest = load_manifest(campaign_dir)
    per_run = []
    counts = {"PASS": 0, "WAIT": 0, "RETRY": 0, "REPAIR": 0, "DIAGNOSE": 0, "AUTO_FAIL": 0}

    for r in results:
        run_id = r.get("experiment_id") or r.get("exp_id") or "unknown"
        config = r.get("config") or {}
        entry = ensure_run(manifest, run_id, tag, config)
        cls = classify_result(r)
        action = cls["next_action"]
        attempt_type = _attempt_type(run_id, state)
        limit_status = check_limit(entry, action, attempt_type, current_classification=cls.get("classification"))
        if limit_status:
            action = limit_status
            cls = {**cls, "next_action": action, "classification": limit_status, "confidence": "high"}
        metrics_path = None
        if r.get("metrics"):
            rd = r.get("remote_results_dir") or r.get("results_dir") or ""
            metrics_path = f"{rd}/metrics.json" if rd else None
        record_attempt(
            manifest=manifest,
            run_id=run_id,
            run_type=tag,
            attempt_type=attempt_type,
            status=r.get("status", "unknown"),
            classification=cls["classification"],
            action=action,
            config=config,
            cluster_id=str(r.get("cluster_id") or "") or None,
            evidence=cls.get("evidence", []),
            metrics_path=metrics_path,
        )
        bucket = action if action in counts else ("AUTO_FAIL" if str(action).startswith("AUTO_FAIL") else action)
        counts[bucket] = counts.get(bucket, 0) + 1
        per_run.append({"run_id": run_id, "arch_name": r.get("arch_name"), "classification": cls["classification"], "action": action, "confidence": cls.get("confidence"), "evidence": cls.get("evidence", []), "missing_evidence": cls.get("missing_evidence", [])})

    active_base_ids = {
        base_run_id(r.get("experiment_id") or r.get("exp_id") or "unknown")
        for r in results
    }
    active_base_ids.update(
        experiment_id(p) for p in state.get("proposals", []) if isinstance(p, dict)
    )
    stale_marked = mark_stale_active_runs(
        manifest,
        active_base_ids=active_base_ids,
        round_prefix=f"r{round_num:03d}_",
        reason="dropped from current round after retry/repair recovery",
    )

    save_manifest(campaign_dir, manifest)

    # Round-level action. Controller is deterministic and conservative.
    if counts.get("WAIT", 0) > 0:
        round_action = "WAIT"
        next_phase = "monitor"
    elif counts.get("DIAGNOSE", 0) > 0:
        round_action = "DIAGNOSE"
        next_phase = "blocked"  # until diagnose action is implemented
    elif counts.get("REPAIR", 0) > 0:
        round_action = "REPAIR"
        next_phase = "smoke_classify"
    elif counts.get("RETRY", 0) > 0:
        round_action = "RETRY"
        next_phase = "submit"  # runner will later specialize retry attempts
    else:
        if tag == "smoke":
            # If at least one passed and all failures are terminal auto-fails, proceed with passed subset.
            round_action = "FULL_SUBMIT" if counts.get("PASS", 0) > 0 else "AUTO_FAIL"
            next_phase = "submit" if counts.get("PASS", 0) > 0 else "blocked"
        else:
            round_action = "ROUND_REVIEW"
            next_phase = "review"

    decision = {
        "ok": True,
        "timestamp": now_iso(),
        "round": round_num,
        "tag": tag,
        "round_action": round_action,
        "next_phase": next_phase,
        "counts": counts,
        "manifest_summary": manifest_summary(manifest),
        "stale_marked": stale_marked,
        "per_run": per_run,
    }
    (art_dir / "controller_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    return decision


def main() -> None:
    campaign_dir = Path(os.environ.get("HYBRID_CAMPAIGN_DIR", "."))
    decision = decide(campaign_dir)
    print(json.dumps({"ok": True, "round_action": decision["round_action"], "next_phase": decision["next_phase"], "counts": decision["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




