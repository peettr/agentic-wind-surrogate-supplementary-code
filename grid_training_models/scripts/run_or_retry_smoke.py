#!/usr/bin/env python3
"""Deterministic dry-run controller for Auto V5 smoke retry/repair decisions."""
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

from scripts.auto_v5_controller import get_base_entry, load_manifest, plan_next_step, save_manifest
from scripts.classify_run_failure import classify_evidence


def decide(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    entry = get_base_entry(manifest, args.base_run_id)
    evidence = json.loads(args.evidence_json.read_text())
    evidence.setdefault("run_id", args.run_id)
    evidence.setdefault("campaign", args.campaign)
    result = classify_evidence(evidence)
    classification = result.to_dict()
    plan = plan_next_step(entry, classification)
    payload = {
        "campaign": args.campaign,
        "base_run_id": args.base_run_id,
        "run_id": args.run_id,
        "classification": classification,
        "plan": plan,
        "dry_run": bool(args.dry_run),
    }
    if args.write_manifest and not args.dry_run:
        save_manifest(args.manifest, manifest)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True)
    p.add_argument("--base-run-id", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--evidence-json", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--write-manifest", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(decide(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
