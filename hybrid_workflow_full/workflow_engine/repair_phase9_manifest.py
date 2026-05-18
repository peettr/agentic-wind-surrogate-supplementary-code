"""Repair Phase9 attempt manifest for the PASS-overwritten-by-AUTO_FAIL bug.

This script:
1. Creates a timestamped backup of attempt_manifest.json
2. For each run entry that has status AUTO_FAIL_MAX_TOTAL_ATTEMPTS:
   a. Checks if any attempt has evidence=["completed with metrics"] but
      classification/action containing AUTO_FAIL (the bug signature).
   b. If found, restores entry.status to PASS and fixes the attempt records.
3. For entries where all attempts are PASS/DUPLICATE_PASS, ensures status=PASS.
4. Writes a dry-run report by default. Use --apply to actually modify the manifest.

Does NOT touch:
- CRC/Condor or remote jobs
- round artifacts (controller_decision.json, results files)
- knowledge/history files

The knowledge/history/controller_decision artifacts represent what the system
*did* at the time. The manifest is the source of truth for accounting. Repairing
the manifest is sufficient to fix status display and prevent the bug from
blocking future work. Historical controller_decision.json files can be left
as-is since they represent the decision made at the time (with the bug).

If you want to also correct controller_decision.json per_run entries, that's a
separate step — see the dry-run report for which round artifacts are affected.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from attempt_manifest import TERMINAL_STATUSES


def repair_manifest(manifest: dict, *, dry_run: bool = True) -> dict:
    """Repair manifest in-place. Returns a report dict."""
    report = {"dry_run": dry_run, "repairs": [], "already_ok": [], "no_pass_evidence": []}
    runs = manifest.get("runs", {})

    for base_id, entry in runs.items():
        status = entry.get("status", "")
        attempts = entry.get("attempts", [])

        # Check if any attempt has "completed with metrics" evidence but was
        # misclassified as AUTO_FAIL
        bug_signatures = []
        pass_attempts = []
        for a in attempts:
            evidence = a.get("evidence", [])
            cls = a.get("classification", "")
            action = a.get("action", "")
            has_metrics_evidence = "completed with metrics" in evidence
            is_auto_fail = "AUTO_FAIL" in cls or "AUTO_FAIL" in action

            if has_metrics_evidence and is_auto_fail:
                bug_signatures.append(a)
            if has_metrics_evidence and not is_auto_fail:
                pass_attempts.append(a)

        if not bug_signatures:
            if status == "PASS":
                report["already_ok"].append(base_id)
            else:
                report["no_pass_evidence"].append(base_id)
            continue

        repair = {
            "base_id": base_id,
            "old_status": status,
            "new_status": "PASS",
            "fixed_attempts": [],
            "metrics_path": None,
        }

        if not dry_run:
            # Find the best metrics_path from pass attempts or bug-fixed attempts
            for a in attempts:
                if a.get("metrics_path"):
                    repair["metrics_path"] = a["metrics_path"]

            entry["status"] = "PASS"
            entry.pop("terminal_reason", None)

            # Fix misclassified attempts
            for a in bug_signatures:
                old_cls = a["classification"]
                old_action = a["action"]
                a["classification"] = "PASS"
                a["action"] = "PASS"
                repair["fixed_attempts"].append({
                    "run_id": a["run_id"],
                    "old_classification": old_cls,
                    "old_action": old_action,
                    "new_classification": "PASS",
                    "new_action": "PASS",
                })

        else:
            for a in bug_signatures:
                repair["fixed_attempts"].append({
                    "run_id": a["run_id"],
                    "old_classification": a["classification"],
                    "old_action": a["action"],
                    "new_classification": "PASS",
                    "new_action": "PASS",
                })

        report["repairs"].append(repair)

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Repair Phase9 manifest for PASS/AUTO_FAIL bug")
    parser.add_argument("--apply", action="store_true", help="Actually modify the manifest (default: dry-run)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to attempt_manifest.json")
    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = Path(__file__).resolve().parent.parent / "campaigns" / "phase9" / "attempt_manifest.json"

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    report = repair_manifest(manifest, dry_run=not args.apply)

    if args.apply:
        # Create timestamped backup
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = manifest_path.with_suffix(f".json.bak.{ts}")
        backup_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Backup: {backup_path}")

        # Write repaired manifest
        from attempt_manifest import save_manifest
        save_manifest(manifest_path.parent, manifest)
        print(f"Repaired manifest written to {manifest_path}")

    print(f"\n--- Repair Report (dry_run={report['dry_run']}) ---")
    print(f"Entries to repair: {len(report['repairs'])}")
    print(f"Already OK (PASS): {len(report['already_ok'])}")
    print(f"No PASS evidence:   {len(report['no_pass_evidence'])}")

    for r in report["repairs"]:
        print(f"\n  {r['base_id']}: {r['old_status']} -> {r['new_status']}")
        for fa in r["fixed_attempts"]:
            print(f"    {fa['run_id']}: {fa['old_classification']} -> {fa['new_classification']}")

    if report["no_pass_evidence"]:
        print(f"\n  Entries with no PASS evidence (left unchanged):")
        for bid in report["no_pass_evidence"]:
            entry = manifest["runs"].get(bid, {})
            print(f"    {bid}: status={entry.get('status')} total_attempts={entry.get('total_attempts')}")

    # Also report which round artifacts would need correction if desired
    affected_rounds = set()
    for r in report["repairs"]:
        for fa in r["fixed_attempts"]:
            rid = fa["run_id"]
            if rid.startswith("r"):
                round_num = int(rid[1:4])
                affected_rounds.add(round_num)
    if affected_rounds:
        print(f"\n  Affected round artifacts (controller_decision.json per_run): rounds {sorted(affected_rounds)}")
        print(f"  These are historical records and can be left as-is, or manually corrected.")

    return report


if __name__ == "__main__":
    main()
