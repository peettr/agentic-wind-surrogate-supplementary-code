#!/usr/bin/env python
"""R13 planner synthesis/gate diagnostic.

Uses existing R13 scout/context artifacts and writes only under
artifacts/r013/planner_synthesis_gate_test_*.

Default mode is parse-only/dry-run against existing artifacts. Use --live to
call workflow_planner.synthesize_proposals and quality_gate_with_codex.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

hybrid_ROOT = Path(__file__).resolve().parents[1]
ENGINE = hybrid_ROOT / "workflow_engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))
if str(hybrid_ROOT) not in sys.path:
    sys.path.insert(0, str(hybrid_ROOT))

import workflow_planner as planner  # noqa: E402

SCOUT_NAMES = ["claude", "codex", "deepseek", "gemini", "grok", "mimo"]
REQUIRED_BASE = [
    "planner_context.json",
    "planner_prompt_claude.txt",
    "planner_prompt_codex.txt",
    "synthesis_claude_prompt.txt",
    "synthesis_claude_raw.txt",
    "synthesis_fallback.json",
    "quality_gate_codex_prompt.txt",
    "quality_gate_codex_raw.txt",
    "quality_gate_codex.json",
    "proposals.json",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_scout_results(base: Path) -> dict[str, dict]:
    scouts: dict[str, dict] = {}
    for name in SCOUT_NAMES:
        p = base / f"scout_{name}.json"
        if p.exists():
            scouts[name] = read_json(p)
    return scouts


def text_flags(text: str) -> dict[str, bool]:
    return {
        "contains_no_further_continuation_needed": "No further continuation needed" in text,
        "contains_markdown_fence": "```" in text,
        "starts_json_object_or_array": text.lstrip().startswith(("{", "[")),
        "nonempty": bool(text.strip()),
    }


def proposal_summary(proposals: list[dict] | None) -> dict[str, Any]:
    proposals = proposals or []
    return {
        "count": len(proposals),
        "arch_names": [p.get("arch_name") for p in proposals[:20]],
        "all_have_required_trainconfig_keys": all(
            isinstance(p, dict)
            and {"arch_name", "n_c", "depth", "loss_name", "lr", "batch_size", "input_features", "epochs", "seed"}.issubset(p)
            for p in proposals
        ),
    }


def inspect_existing(base: Path) -> dict[str, Any]:
    scouts = load_scout_results(base)
    syn_raw_path = base / "synthesis_claude_raw.txt"
    gate_raw_path = base / "quality_gate_codex_raw.txt"
    syn_raw = syn_raw_path.read_text(encoding="utf-8", errors="replace") if syn_raw_path.exists() else ""
    gate_raw = gate_raw_path.read_text(encoding="utf-8", errors="replace") if gate_raw_path.exists() else ""
    syn_wrapper = planner.extract_proposal_wrapper(syn_raw)
    syn_props = syn_wrapper.get("proposals") if syn_wrapper else planner.extract_proposals(syn_raw)
    gate_obj = None
    gate_props = None
    if gate_raw.strip():
        try:
            gate_obj = json.loads(gate_raw)
            if isinstance(gate_obj, dict):
                gate_props = gate_obj.get("proposals")
        except json.JSONDecodeError:
            gate_props = planner.extract_proposals(gate_raw)
    return {
        "artifact_dir": str(base),
        "required_artifacts": {name: (base / name).exists() for name in REQUIRED_BASE},
        "scouts": {
            k: {
                "status": v.get("status"),
                "model": v.get("model"),
                "proposal_count": len(v.get("proposals") or []),
                "auxiliary_ideas_count": len(v.get("auxiliary_ideas") or []),
                "raw_text_len": len(v.get("raw_text") or ""),
            }
            for k, v in scouts.items()
        },
        "existing_synthesis_raw": {
            "path": str(syn_raw_path),
            "size": syn_raw_path.stat().st_size if syn_raw_path.exists() else 0,
            **text_flags(syn_raw),
            "extract_wrapper_ok": bool(syn_wrapper),
            "extract_proposals_ok": bool(syn_props),
            "proposal_summary": proposal_summary(syn_props),
        },
        "existing_gate_raw": {
            "path": str(gate_raw_path),
            "size": gate_raw_path.stat().st_size if gate_raw_path.exists() else 0,
            **text_flags(gate_raw),
            "json_object_ok": isinstance(gate_obj, dict),
            "extract_proposals_ok": bool(gate_props),
            "proposal_summary": proposal_summary(gate_props),
        },
    }


def run_live(base: Path, out_dir: Path, target_count: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    context = read_json(base / "planner_context.json")
    # synthesize_proposals points the prompt at art_dir/planner_context.json.
    shutil.copy2(base / "planner_context.json", out_dir / "planner_context.json")
    scouts = load_scout_results(base)
    started = time.time()
    synth_props = planner.synthesize_proposals(scouts, context, target_count, out_dir)
    after_synth = time.time()
    gate_props = planner.quality_gate_with_codex(synth_props, context, target_count, out_dir)
    ended = time.time()

    syn_raw = (out_dir / "synthesis_claude_raw.txt").read_text(encoding="utf-8", errors="replace") if (out_dir / "synthesis_claude_raw.txt").exists() else ""
    gate_raw = (out_dir / "quality_gate_codex_raw.txt").read_text(encoding="utf-8", errors="replace") if (out_dir / "quality_gate_codex_raw.txt").exists() else ""
    syn_wrapper = planner.extract_proposal_wrapper(syn_raw)
    syn_extracted = syn_wrapper.get("proposals") if syn_wrapper else planner.extract_proposals(syn_raw)
    gate_json_ok = False
    gate_parsed_props = None
    if gate_raw.strip():
        try:
            gate_obj = json.loads(gate_raw)
            gate_json_ok = isinstance(gate_obj, dict)
            if isinstance(gate_obj, dict):
                gate_parsed_props = gate_obj.get("proposals")
        except json.JSONDecodeError:
            gate_parsed_props = planner.extract_proposals(gate_raw)

    return {
        "output_dir": str(out_dir),
        "durations_sec": {"synthesis": round(after_synth - started, 2), "gate": round(ended - after_synth, 2), "total": round(ended - started, 2)},
        "synthesis": {
            "returned": proposal_summary(synth_props),
            "raw": {**text_flags(syn_raw), "size": len(syn_raw), "extract_wrapper_ok": bool(syn_wrapper), "extract_proposals_ok": bool(syn_extracted), "extracted": proposal_summary(syn_extracted)},
            "fallback_triggered": (out_dir / "synthesis_fallback.json").exists() and not (out_dir / "synthesis_claude.json").exists(),
            "artifacts": {name: (out_dir / name).exists() for name in ["synthesis_claude_prompt.txt", "synthesis_claude_raw.txt", "synthesis_claude.json", "synthesis_fallback.json", "synthesis_claude_error.txt"]},
        },
        "codex_gate": {
            "returned": proposal_summary(gate_props),
            "raw": {**text_flags(gate_raw), "size": len(gate_raw), "json_object_ok": gate_json_ok, "extract_proposals_ok": bool(gate_parsed_props), "extracted": proposal_summary(gate_parsed_props)},
            "fell_back_to_synthesis_output": gate_props == synth_props[:target_count] and not (out_dir / "quality_gate_codex.json").exists(),
            "artifacts": {name: (out_dir / name).exists() for name in ["quality_gate_codex_prompt.txt", "quality_gate_codex_raw.txt", "quality_gate_codex_terminal.txt", "quality_gate_codex.json", "quality_gate_codex_error.txt"]},
        },
    }


def write_md(path: Path, summary: dict[str, Any]) -> None:
    live = summary.get("live")
    lines = ["# R13 planner synthesis/gate test", "", f"Mode: {'live' if live else 'parse-only'}", ""]
    ex = summary["existing"]
    lines += [
        "## Existing artifacts",
        f"- Scouts loaded: {', '.join(ex['scouts'].keys())}",
        f"- Existing Claude raw parseable: {ex['existing_synthesis_raw']['extract_proposals_ok']} (count={ex['existing_synthesis_raw']['proposal_summary']['count']})",
        f"- Existing Claude raw flags: no_continuation={ex['existing_synthesis_raw']['contains_no_further_continuation_needed']}, fence={ex['existing_synthesis_raw']['contains_markdown_fence']}, starts_json={ex['existing_synthesis_raw']['starts_json_object_or_array']}",
        f"- Existing Codex gate parseable: {ex['existing_gate_raw']['extract_proposals_ok']} (count={ex['existing_gate_raw']['proposal_summary']['count']})",
        "",
    ]
    if live:
        lines += [
            "## Live run",
            f"- Output dir: `{live['output_dir']}`",
            f"- Durations: {live['durations_sec']}",
            f"- Claude returned proposals: {live['synthesis']['returned']['count']}",
            f"- Claude raw parseable: {live['synthesis']['raw']['extract_proposals_ok']} / wrapper={live['synthesis']['raw']['extract_wrapper_ok']}",
            f"- Claude raw flags: no_continuation={live['synthesis']['raw']['contains_no_further_continuation_needed']}, fence={live['synthesis']['raw']['contains_markdown_fence']}, starts_json={live['synthesis']['raw']['starts_json_object_or_array']}",
            f"- Synthesis fallback triggered: {live['synthesis']['fallback_triggered']}",
            f"- Codex returned proposals: {live['codex_gate']['returned']['count']}",
            f"- Codex raw JSON object ok: {live['codex_gate']['raw']['json_object_ok']}; extract_proposals_ok={live['codex_gate']['raw']['extract_proposals_ok']}",
            f"- Codex fell back to synthesis output: {live['codex_gate']['fell_back_to_synthesis_output']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=hybrid_ROOT / "campaigns" / "hybrid" / "artifacts" / "r013")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--target-count", type=int, default=12)
    ap.add_argument("--live", action="store_true", help="Actually call Claude synthesis and Codex gate")
    args = ap.parse_args()

    base = args.base.resolve()
    if not base.exists():
        raise SystemExit(f"missing base artifact dir: {base}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out_dir or (base / f"planner_synthesis_gate_test_{stamp}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "timestamp": stamp,
        "mode": "live" if args.live else "parse-only",
        "script": str(Path(__file__).resolve()),
        "base": str(base),
        "existing": inspect_existing(base),
    }
    if args.live:
        summary["live"] = run_live(base, out_dir, args.target_count)

    summary_path = out_dir / "summary.json"
    md_path = out_dir / "summary.md"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(md_path, summary)
    print(json.dumps({"summary_json": str(summary_path), "summary_md": str(md_path), "mode": summary["mode"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



