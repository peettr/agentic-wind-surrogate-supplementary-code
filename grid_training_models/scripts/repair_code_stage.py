#!/usr/bin/env python3
"""Build and optionally execute the Grid repair stage.

The mandated repair sequence is Claude repair -> Codex review+patch -> Claude
final review+patch. By default this module is dry-run friendly: it writes the
context and command plan without executing AI tools.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_repair_context(context: dict[str, Any]) -> str:
    lines: list[str] = ["# Grid Repair Context", ""]
    scalar_keys = [
        "base_run_id",
        "run_id",
        "campaign",
        "classification",
        "model_file",
        "train_config",
        "retry_count",
        "repair_count",
        "total_attempts",
    ]
    for key in scalar_keys:
        if key in context:
            lines.append(f"{key}: {context[key]}")
    lines.extend([
        "",
        "## Repair sequence",
        "1. Claude CLI repair",
        "2. Codex CLI review + patch",
        "3. Claude CLI final review + patch",
        "",
        "## Limits and constraints",
        "- Do not lower smoke batch_size below 8.",
        "- Do not change smoke epochs away from 20.",
        "- Do not change seed, input_features, compute_r2, or eval_splits unless explicitly instructed.",
        "- Keep retry/repair bookkeeping out of train_config.json.",
    ])
    if context.get("limits"):
        lines.extend(["", "## Limits", "```json", json.dumps(context["limits"], indent=2), "```"])
    if context.get("forbidden_changes"):
        lines.extend(["", "## Forbidden changes"])
        lines.extend(f"- {item}" for item in context["forbidden_changes"])
    if context.get("allowed_change_paths"):
        lines.extend(["", "## Allowed change paths"])
        lines.extend(f"- {item}" for item in context["allowed_change_paths"] if item)
    if context.get("forbidden_change_paths"):
        lines.extend(["", "## Forbidden change paths"])
        lines.extend(f"- {item}" for item in context["forbidden_change_paths"] if item)
    if context.get("attempts"):
        lines.extend(["", "## Previous attempts", "```json", json.dumps(context["attempts"], indent=2), "```"])
    if context.get("logs"):
        lines.extend(["", "## Logs"])
        for name, text in context["logs"].items():
            lines.extend([f"### {name}", "```text", str(text), "```"])
    if context.get("controller_state_entry"):
        lines.extend(["", "## Controller state entry", "```json", json.dumps(context["controller_state_entry"], indent=2, sort_keys=True), "```"])
    if context.get("repair_run_entry"):
        lines.extend(["", "## Repair run entry", "```json", json.dumps(context["repair_run_entry"], indent=2, sort_keys=True), "```"])
    if context.get("source_control_row"):
        lines.extend(["", "## Source control row", "```json", json.dumps(context["source_control_row"], indent=2, sort_keys=True), "```"])
    if context.get("file_snapshots"):
        lines.extend(["", "## Relevant file snapshots"])
        for item in context["file_snapshots"]:
            lines.extend([f"### File: {item['path']}", "```" + item.get("language", "text"), item.get("content", ""), "```"])
    return "\n".join(lines) + "\n"


def build_repair_plan(context: dict[str, Any], *, repair_id: str, output_dir: Path) -> dict[str, Any]:
    context_path = output_dir / repair_id / "context.md"
    prompt = (
        f"Read {context_path} and perform the requested Grid repair/review stage. "
        "Within the current hard constraints and allowed change paths, autonomously revise allowed change paths as needed; "
        "do not ask for approval for changes inside allowed change paths. Preserve experiment contracts."
    )
    return {
        "repair_id": repair_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "context_path": str(context_path),
        "steps": [
            {
                "agent": "claude",
                "role": "repair",
                "command": ["claude", "--print", prompt],
            },
            {
                "agent": "codex",
                "role": "review_patch",
                "command": ["<LOCAL_HOME_PATH>", "exec", "--skip-git-repo-check", prompt],
            },
            {
                "agent": "claude",
                "role": "final_review_patch",
                "command": ["claude", "--print", prompt],
            },
        ],
    }


def write_repair_stage(context: dict[str, Any], *, repair_id: str, output_root: Path, execute: bool = False) -> dict[str, Any]:
    repair_dir = output_root / repair_id
    repair_dir.mkdir(parents=True, exist_ok=True)
    context_text = build_repair_context(context)
    (repair_dir / "context.md").write_text(context_text)
    plan = build_repair_plan(context, repair_id=repair_id, output_dir=output_root)
    (repair_dir / "commands.json").write_text(json.dumps(plan, indent=2) + "\n")
    if execute:
        raise NotImplementedError("live AI repair execution is intentionally gated for a later explicit implementation")
    return {"repair_id": repair_id, "repair_dir": str(repair_dir), "dry_run": True, "plan": plan}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context-json", type=Path, required=True)
    p.add_argument("--repair-id", required=True)
    p.add_argument("--output-root", type=Path, default=Path("reports/repairs"))
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    context = json.loads(args.context_json.read_text())
    result = write_repair_stage(context, repair_id=args.repair_id, output_root=args.output_root, execute=args.execute)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



