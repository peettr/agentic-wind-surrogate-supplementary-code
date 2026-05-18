"""Sequential Workflow Reviewer â€” Dual-model analysis for smoke and full runs.

Two modes:
1. smoke_classify: diagnose failures, produce fix plan (Sonnet + Codex)
2. review: analyze full results, produce round review (Opus + Codex)

Both modes use dual-model calling with synthesis, following V1's proven pattern.

Called by runner during 'smoke_classify' and 'review' phases.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "workflow_engine"))
sys.path.insert(0, str(PROJECT_ROOT / "explorer"))

from workflow_common import (
    now_iso, load_state, round_artifact_dir, history_path,
)
from workflow_knowledge import build_experiment_knowledge

CLAUDE_BIN_FULL = shutil.which("claude") or shutil.which("claude.exe") or "claude"
CODEX_BIN_FULL = shutil.which("codex") or "codex"
CODEX_MODEL = os.environ.get("hybrid_REVIEW_CODEX_MODEL", "gpt-5.5")


def run_cmd(cmd: list[str], timeout: int = 900, stdin_text: str | None = None,
            cwd: Path | None = None) -> tuple[bool, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       timeout=timeout, input=stdin_text, cwd=str(cwd) if cwd else None)
    text = ((r.stdout or "") + ("\n" + r.stderr if r.stderr else "")).strip()
    return r.returncode == 0, text


# â”€â”€ Post-codegen review â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _proposal_by_arch(state: dict) -> dict[str, dict]:
    return {p.get("arch_name", ""): p for p in state.get("proposals", []) if p.get("arch_name")}


def build_post_codegen_prompt(state: dict, manifest: dict, target_archs: list[str]) -> str:
    proposals = _proposal_by_arch(state)
    payload = []
    for arch in target_archs:
        path = PROJECT_ROOT / "models" / "generated" / f"{arch}.py"
        shared_path = PROJECT_ROOT / "shared" / "models" / f"{arch}.py"
        code = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        payload.append({
            "arch_name": arch,
            "generated_path": str(path),
            "shared_model_exists": shared_path.exists(),
            "proposal": proposals.get(arch, {}),
            "code_excerpt": code[:5000],
        })

    train_contract = (
        "Important Sequential loading contract: TrainConfig may include script_path. "
        "hybrid/shared/train.py must prefer script_path over shared MODEL_REGISTRY "
        "when script_path is present, so generated/fixed files are actually used. "
        "A generated file must be self-contained: all helper classes/functions it references must be defined in that file. "
        "Each model class must be named exactly arch_name and accept __init__(in_channels=1, out_channels=1, n_c=..., depth=...)."
    )
    return (
        "You are performing post-codegen review for Hybrid neural surrogate models.\n"
        "You MAY directly edit files under hybrid/models/generated only.\n"
        "Do not edit locked infrastructure: hybrid/shared/train.py, losses.py, eval_module.py, data loader, data files, or templates.\n\n"
        f"{train_contract}\n\n"
        "Review goals:\n"
        "1. Fix undefined symbols and missing helper classes.\n"
        "2. Ensure generated/fixed code will actually be loaded by train.py.\n"
        "3. Ensure arch_kwargs/config match model __init__.\n"
        "4. Keep fixes minimal and runnable on 640x640 tensors.\n"
        "5. Preserve NaN masking contract and avoid nan_to_num.\n\n"
        f"Codegen manifest:\n{json.dumps(manifest, indent=2, ensure_ascii=False)}\n\n"
        f"Targets:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        "After editing, print a compact JSON object with keys: action, changed_files, notes."
    )


def call_claude_post_codegen(prompt: str, art_dir: Path, stage: str) -> tuple[bool, str]:
    cache = art_dir / f"post_codegen_{stage}_claude.txt"
    ok, out = run_cmd(
        [CLAUDE_BIN_FULL, "--model", "opus", "--permission-mode", "bypassPermissions",
         "--dangerously-skip-permissions", "--print"],
        timeout=900, stdin_text=prompt, cwd=PROJECT_ROOT,
    )
    cache.write_text(out or "", encoding="utf-8")
    return ok, out


def call_codex_post_codegen(prompt: str, art_dir: Path) -> tuple[bool, str]:
    cache = art_dir / "post_codegen_codex.txt"
    ok, out = run_cmd(
        [CODEX_BIN_FULL, "exec", "--model", CODEX_MODEL, "--skip-git-repo-check", "--cd", str(PROJECT_ROOT)],
        timeout=900, stdin_text=prompt, cwd=PROJECT_ROOT,
    )
    cache.write_text(out or "", encoding="utf-8")
    return ok, out


def _manifest_blocking_failures(manifest: dict) -> list[str]:
    failures: list[str] = []
    for key in (
        "failed", "failed_run_ids", "failed_details",
        "failed_skipped_after_codegen_retries", "failed_terminal_after_codegen_limits",
        "terminal_codegen_failures",
    ):
        val = manifest.get(key)
        if not val:
            continue
        if isinstance(val, list):
            failures.extend(f"{key}: {item}" for item in val)
        else:
            failures.append(f"{key}: {val}")
    return failures


def run_post_codegen_validation(state: dict, manifest: dict, target_archs: list[str]) -> dict:
    from workflow_codegen import validate_model_code

    proposals = _proposal_by_arch(state)
    validated = []
    failed = []
    for arch in target_archs:
        cfg = proposals.get(arch, {"arch_name": arch})
        ok, msg = validate_model_code(arch, cfg)
        if ok:
            validated.append(arch)
        else:
            failed.append(f"{arch}: {msg}")
    manifest_failures = _manifest_blocking_failures(manifest)
    failed.extend(manifest_failures)
    all_target_archs_valid = len(validated) == len(target_archs)
    return {
        "ok": all_target_archs_valid and len(failed) == 0,
        "reviewed_archs": target_archs,
        "validated_archs": validated,
        "failed": failed,
        "manifest_failures": manifest_failures,
        "manifest_validated_archs": manifest.get("validated_archs", []),
    }


# â”€â”€ Smoke diagnosis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def build_smoke_diagnosis_prompt(smoke_results: list[dict]) -> str:
    failures = [r for r in smoke_results if r["status"] != "completed"]
    if not failures:
        return ""
    return (
        "You are diagnosing smoke test failures for neural network training runs.\n"
        "Each experiment ran for 20 epochs (smoke test) on 640x640 wind pressure prediction.\n\n"
        "Failed experiments:\n"
        f"{json.dumps(failures, indent=2, ensure_ascii=False)}\n\n"
        "For each failure, diagnose:\n"
        "1. Root cause (OOM, shape mismatch, NaN, import error, etc.)\n"
        "2. Whether it's fixable by code modification\n"
        "3. Specific fix needed (be concrete: change n_c from 32 to 16, add try/except, etc.)\n\n"
        "Use fix_type=config only for safe capacity/training knob changes such as OOM, batch, depth, or lr. "
        "Use fix_type=schema for constructor/config interface mismatches such as unexpected keyword arguments, "
        "where a kwarg must be renamed, removed, or mapped to the model's actual __init__ signature. "
        "Use fix_type=code for model implementation bugs.\n\n"
        "Return JSON:\n"
        '{"fixes": [{"exp_id": "...", "arch_name": "...", "diagnosis": "...", '
        '"fixable": true, "fix_description": "...", "fix_type": "code|config|schema"}]}\n'
    )


def call_claude_smoke(prompt: str, art_dir: Path) -> dict | None:
    """Call Claude Sonnet for smoke diagnosis."""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    cache = art_dir / f"smoke_diagnosis_claude_{prompt_hash}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    ok, out = run_cmd(
        [CLAUDE_BIN_FULL, "--model", "sonnet", "--permission-mode",
         "bypassPermissions", "--print"],
        timeout=600, stdin_text=prompt,
    )
    if ok and out:
        result = extract_json_with_key(out, "fixes")
        if result:
            cache.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result
    return None


def call_codex_smoke(prompt: str, art_dir: Path) -> dict | None:
    """Call Codex for smoke diagnosis."""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    cache = art_dir / f"smoke_diagnosis_codex_{prompt_hash}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    ok, out = run_cmd(
        [CODEX_BIN_FULL, "exec", "--model", CODEX_MODEL, "--skip-git-repo-check",
         "--cd", "/tmp", "--ephemeral", "--ignore-rules"],
        timeout=600, stdin_text=prompt,
    )
    if ok and out:
        result = extract_json_with_key(out, "fixes")
        if result:
            cache.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result
    return None


def synthesize_smoke_diagnosis(claude_result: dict | None, codex_result: dict | None) -> dict:
    """Merge dual-model smoke diagnoses."""
    claude_fixes = (claude_result or {}).get("fixes", [])
    codex_fixes = (codex_result or {}).get("fixes", [])

    # Merge by exp_id: if both agree on fixable, use more detailed diagnosis
    merged = {}
    for f in claude_fixes + codex_fixes:
        eid = f.get("exp_id", "")
        if eid not in merged:
            merged[eid] = f
        else:
            existing = merged[eid]
            # Prefer fixable=True diagnosis (more actionable)
            if f.get("fixable") and not existing.get("fixable"):
                merged[eid] = f
            elif f.get("fixable") == existing.get("fixable"):
                # Same verdict â€” keep longer fix_description
                if len(f.get("fix_description", "")) > len(existing.get("fix_description", "")):
                    merged[eid] = f

    fixes = list(merged.values())
    return {
        "has_fixable": any(f.get("fixable") for f in fixes),
        "fixes": fixes,
        "claude_fixes": len(claude_fixes),
        "codex_fixes": len(codex_fixes),
    }


def _safe_config_patch(failure: dict, fix: dict) -> dict:
    """Return a conservative patch for config-level repairs.

    Only training/capacity knobs are allowed.  Data, seed, split, and feature
    choices are intentionally immutable here.
    """
    cfg = failure.get("config") or {}
    text = "\n".join([
        str(fix.get("diagnosis", "")),
        str(fix.get("fix_description", "")),
        str(failure.get("log_tail", "")),
        str(failure.get("error", "")),
    ]).lower()
    patch: dict = {}

    cur_batch = int(cfg.get("batch_size") or 16)
    cur_nc = int(cfg.get("n_c") or (cfg.get("arch_kwargs") or {}).get("n_c") or 16)
    cur_depth = int(cfg.get("depth") or (cfg.get("arch_kwargs") or {}).get("depth") or 7)
    cur_lr = float(cfg.get("lr") or 1.0e-3)

    # Explicit text like "n_c from 32 to 16" wins.
    m = re.search(r"n_c\s*(?:from|=)?\s*(\d+)\s*(?:to|->)\s*(\d+)", text)
    if m:
        patch["n_c"] = max(8, min(cur_nc, int(m.group(2))))
    elif any(k in text for k in ("oom", "out of memory", "too large", "memory")):
        if cur_nc >= 32:
            patch["n_c"] = 16
        elif cur_nc >= 24:
            patch["n_c"] = 16

    if any(k in text for k in ("oom", "out of memory", "memory", "batch")):
        # Automatic OOM/config repair may reduce batch_size only to the Hybrid
        # floor of 8.  batch_size<8 is reserved for explicit manual_resource_probe_approved
        # feasibility probes and must not be suggested/submitted by autonomous repair.
        if cur_batch > 8:
            patch["batch_size"] = 8

    if re.search(r"(?:reduce|drop|lower)\s+depth|depth\s*(?:from|=)?\s*\d+\s*(?:to|->)\s*\d+", text) and cur_depth > 5:
        m_depth = re.search(r"depth\s*(?:from|=)?\s*(\d+)\s*(?:to|->)\s*(\d+)", text)
        patch.setdefault("depth", max(5, min(cur_depth, int(m_depth.group(2)) if m_depth else cur_depth - 1)))

    if "loss_nan" in text or "nan" in text:
        patch["lr"] = max(cur_lr * 0.5, 1.0e-5)

    allowed = {"n_c", "depth", "batch_size", "lr"}
    return {k: v for k, v in patch.items() if k in allowed}


def _model_init_params(arch_name: str) -> set[str]:
    """Best-effort parse of a shared model __init__ signature."""
    import ast

    model_path = PROJECT_ROOT / "shared" / "models" / f"{arch_name}.py"
    if not model_path.exists():
        return set()
    try:
        tree = ast.parse(model_path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    wanted = {arch_name, arch_name.upper(), arch_name.lower()}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in wanted:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return {a.arg for a in item.args.args if a.arg != "self"}
    return set()


def _schema_config_patch(failure: dict, fix: dict) -> dict:
    """Return a constructor/schema patch for config-interface mismatches.

    This is intentionally separate from safe capacity patches.  It only changes
    how arch_kwargs are passed to the model constructor; it does not change the
    experiment's data, split, seed, loss, or training budget.
    """
    cfg = failure.get("config") or {}
    arch_name = fix.get("arch_name") or cfg.get("arch_name", "")
    text = "\n".join([
        str(fix.get("diagnosis", "")),
        str(fix.get("fix_description", "")),
        str(failure.get("log_tail", "")),
        str(failure.get("error", "")),
    ])
    m = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", text)
    if not m:
        return {}

    bad_kw = m.group(1)
    params = _model_init_params(str(arch_name))
    patch: dict = {"arch_kwargs_remove": [bad_kw], "arch_kwargs_set": {}}

    # Common Sequential convention: n_c is the search-space channel width.  Some
    # imported architectures call the same concept width, hidden_channels, etc.
    if bad_kw == "n_c":
        candidates = [
            "width", "hidden_channels", "channels", "base_channels",
            "dim", "embed_dim", "features", "num_channels",
        ]
        target = next((name for name in candidates if name in params), None)
        if target:
            patch["arch_kwargs_set"][target] = cfg.get("n_c") or (cfg.get("arch_kwargs") or {}).get("n_c", 16)

    # Keep depth when the model accepts it.  The submit layer passes depth by
    # default, so no explicit mapping is needed unless a future model renames it.
    if not patch["arch_kwargs_set"] and bad_kw == "n_c":
        return {}
    return patch


def attach_config_patches(fix_plan: dict, smoke_results: list[dict]) -> dict:
    """Add structured config/schema patch fields to reviewer fixes."""
    by_id = {
        (r.get("experiment_id") or r.get("exp_id")): r
        for r in smoke_results
        if r.get("experiment_id") or r.get("exp_id")
    }
    for fix in fix_plan.get("fixes", []):
        if not fix.get("fixable"):
            continue
        failure = by_id.get(fix.get("exp_id", "")) or {}
        text = "\n".join([
            str(fix.get("diagnosis", "")),
            str(fix.get("fix_description", "")),
            str(failure.get("log_tail", "")),
            str(failure.get("error", "")),
        ]).lower()
        fix_type = str(fix.get("fix_type", "code")).lower()

        if "unexpected keyword argument" in text or fix_type == "schema":
            patch = _schema_config_patch(failure, fix)
            if patch:
                fix["fix_type"] = "schema"
                fix["schema_patch"] = patch
                fix["patch_policy"] = "constructor_schema_only"
            else:
                # Schema mismatch was detected but cannot be expressed safely as
                # a config transform. Route to code repair instead of pretending
                # a capacity patch can fix it.
                fix["fix_type"] = "code"
            continue

        if "config" in fix_type:
            patch = _safe_config_patch(failure, fix)
            if patch:
                fix["config_patch"] = patch
                fix["patch_policy"] = "safe_capacity_only"
            else:
                fix["fixable"] = False
                fix["unfixable_reason"] = "config fix requested but no safe capacity patch could be derived"
    fix_plan["has_fixable"] = any(f.get("fixable") for f in fix_plan.get("fixes", []))
    return fix_plan


# â”€â”€ Full review â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _round_from_record(record: dict) -> int | None:
    """Best-effort deterministic round extraction from record fields."""
    for key in ("round", "round_num"):
        value = record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    for key in ("exp_id", "experiment_id"):
        value = str(record.get(key, ""))
        m = re.search(r"r(\d{3})", value)
        if m:
            return int(m.group(1))
    cfg = record.get("config") if isinstance(record.get("config"), dict) else {}
    value = str(cfg.get("experiment_id", ""))
    m = re.search(r"r(\d{3})", value)
    return int(m.group(1)) if m else None


def _metric_float(record: dict, *paths: tuple[str, ...] | str) -> float | None:
    """Read the first numeric metric from a set of dotted/key paths."""
    for path in paths:
        parts = path.split(".") if isinstance(path, str) else list(path)
        value = record
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    return None


def _full_record_brief(record: dict) -> dict:
    """Compact full-run record for review context tables."""
    cfg = record.get("config") if isinstance(record.get("config"), dict) else {}
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    return {
        "exp_id": record.get("exp_id") or record.get("experiment_id") or metrics.get("experiment_id"),
        "round": _round_from_record(record),
        "arch": record.get("arch_name") or cfg.get("arch_name") or metrics.get("arch_name"),
        "status": record.get("status") or metrics.get("status"),
        "failure_class": record.get("failure_class") or record.get("auto_failure_class"),
        "val_r2_median": _metric_float(record, "val_r2_median", "metrics.val_metrics.r2_median"),
        "val_r2_mean": _metric_float(record, "val_r2_mean", "metrics.val_metrics.r2_mean"),
        "config": {
            "n_c": record.get("n_c") or cfg.get("n_c"),
            "depth": record.get("depth") or cfg.get("depth"),
            "lr": record.get("lr") or cfg.get("lr"),
            "loss_name": record.get("loss_name") or cfg.get("loss_name") or metrics.get("loss_name"),
            "batch_size": record.get("batch_size") or cfg.get("batch_size"),
            "input_features": record.get("input_features") or cfg.get("input_features"),
            "use_ema": cfg.get("use_ema"),
        },
    }


def _is_full_completed(record: dict) -> bool:
    exp_id = str(record.get("exp_id") or record.get("experiment_id") or "")
    status = str(record.get("status") or "").lower()
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    metrics_status = str(metrics.get("status") or "").lower()
    has_r2 = _metric_float(record, "val_r2_median", "metrics.val_metrics.r2_median") is not None
    return "full" in exp_id and (status == "completed" or metrics_status == "ok" or has_r2)



def _campaign_dir_from_env() -> Path:
    return Path(os.environ.get("HYBRID_CAMPAIGN_DIR", PROJECT_ROOT / "campaigns" / "hybrid"))


def _record_exp_id(record: dict) -> str:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    return str(record.get("exp_id") or record.get("experiment_id") or metrics.get("experiment_id") or "")


def _semantic_signature(record: dict) -> tuple:
    cfg = record.get("config") if isinstance(record.get("config"), dict) else {}
    return (
        record.get("arch_name") or cfg.get("arch_name"),
        record.get("n_c") or cfg.get("n_c"),
        record.get("depth") or cfg.get("depth"),
        record.get("input_features") or cfg.get("input_features"),
        record.get("lr") or cfg.get("lr"),
        record.get("loss_name") or cfg.get("loss_name"),
        cfg.get("augmentation") or record.get("augmentation"),
        cfg.get("use_ema") if "use_ema" in cfg else record.get("use_ema"),
    )


def _semantic_signature_from_brief(item: dict) -> tuple:
    cfg = item.get("config") if isinstance(item.get("config"), dict) else {}
    return (
        item.get("arch_name") or item.get("arch") or cfg.get("arch_name"),
        item.get("n_c") or cfg.get("n_c"),
        item.get("depth") or cfg.get("depth"),
        item.get("input_features") or cfg.get("input_features"),
        item.get("lr") or cfg.get("lr"),
        item.get("loss_name") or cfg.get("loss_name"),
        item.get("augmentation") or cfg.get("augmentation"),
        item.get("use_ema") if "use_ema" in item else cfg.get("use_ema"),
    )


def _dedupe_scored_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep best scored record per exp_id and semantic config; return duplicate audit groups."""
    def score(r: dict) -> float:
        return _metric_float(r, "val_r2_median", "metrics.val_metrics.r2_median") or -999.0

    by_exp: dict[str, list[dict]] = {}
    for r in records:
        eid = _record_exp_id(r) or f"id:{id(r)}"
        by_exp.setdefault(eid, []).append(r)
    exp_unique: list[dict] = []
    duplicate_groups: list[dict] = []
    for eid, group in by_exp.items():
        group_sorted = sorted(group, key=score, reverse=True)
        exp_unique.append(group_sorted[0])
        if len(group) > 1:
            duplicate_groups.append({
                "kind": "exp_id",
                "key": eid,
                "kept": _record_exp_id(group_sorted[0]),
                "members": [_record_exp_id(x) for x in group],
            })

    by_sig: dict[tuple, list[dict]] = {}
    for r in exp_unique:
        by_sig.setdefault(_semantic_signature(r), []).append(r)
    semantic_unique: list[dict] = []
    for sig, group in by_sig.items():
        group_sorted = sorted(group, key=score, reverse=True)
        semantic_unique.append(group_sorted[0])
        if len(group) > 1:
            duplicate_groups.append({
                "kind": "semantic_signature",
                "key": list(sig),
                "kept": _record_exp_id(group_sorted[0]),
                "members": [_record_exp_id(x) for x in group],
            })
    semantic_unique.sort(key=score, reverse=True)
    return semantic_unique, duplicate_groups


def _load_round_full_results(campaign_dir: Path, rnd: int, fallback: list[dict] | None = None) -> tuple[list[dict], Path | None]:
    path = campaign_dir / "artifacts" / f"r{rnd:03d}" / "full_results.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data, path
        except (OSError, json.JSONDecodeError):
            return [], path
    return (fallback or []), None


def build_recent_rounds_summary(history: list[dict], full_results: list[dict], max_rounds: int = 5) -> list[dict]:
    """Summarize recent rounds strictly from artifacts/rXXX/full_results.json source files."""
    campaign_dir = _campaign_dir_from_env()
    artifacts_dir = campaign_dir / "artifacts"
    rounds: set[int] = set()
    if artifacts_dir.exists():
        for path in artifacts_dir.glob("r*/full_results.json"):
            m = re.search(r"r(\d{3})", str(path.parent.name))
            if m:
                rounds.add(int(m.group(1)))
    current_rounds = {_round_from_record(r) for r in full_results}
    rounds.update(r for r in current_rounds if r is not None)

    summaries = []
    for rnd in sorted(rounds)[-max_rounds:]:
        fallback = full_results if rnd in current_rounds else []
        records, source_path = _load_round_full_results(campaign_dir, rnd, fallback=fallback)
        completed = [r for r in records if str(r.get("status") or "").lower() == "completed"]
        failed = [r for r in records if str(r.get("status") or "").lower() == "failed"]
        scored = [r for r in completed if _metric_float(r, "val_r2_median", "metrics.val_metrics.r2_median") is not None]
        scored_unique, duplicate_groups = _dedupe_scored_records(scored)
        statuses: dict[str, int] = {}
        failure_classes: dict[str, int] = {}
        for r in records:
            metrics = r.get("metrics") if isinstance(r.get("metrics"), dict) else {}
            status = str(r.get("status") or metrics.get("status") or "UNKNOWN")
            statuses[status] = statuses.get(status, 0) + 1
            cls = r.get("failure_class") or r.get("auto_failure_class")
            if cls:
                failure_classes[str(cls)] = failure_classes.get(str(cls), 0) + 1
        top3_vals = [_metric_float(r, "val_r2_median", "metrics.val_metrics.r2_median") for r in scored_unique[:3]]
        top3_vals = [v for v in top3_vals if v is not None]
        summaries.append({
            "round": rnd,
            "source_file": str(source_path) if source_path else "fallback_current_full_results",
            "source_record_count": len(records),
            "total_full_records": len(records),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "scored_count": len(scored_unique),
            "status_counts": statuses,
            "failure_class_counts": failure_classes,
            "PASS": len(scored_unique),
            "AUTO_FAIL": failure_classes.get("AUTO_FAIL", 0) + statuses.get("AUTO_FAIL", 0),
            "best": _full_record_brief(scored_unique[0]) if scored_unique else None,
            "top3_mean_val_r2_median": (sum(top3_vals) / len(top3_vals)) if top3_vals else None,
            "top3": [_full_record_brief(r) for r in scored_unique[:3]],
            "top3_unique": [_full_record_brief(r) for r in scored_unique[:3]],
            "top5_unique": [_full_record_brief(r) for r in scored_unique[:5]],
            "duplicate_groups": duplicate_groups,
        })
    return summaries

def build_current_best_anchor(state: dict, history: list[dict], full_results: list[dict]) -> dict:
    """Find all-time best/current anchor from known full records and state fallback."""
    candidates = [r for r in [*history, *full_results] if _is_full_completed(r)]
    candidates = [r for r in candidates if _metric_float(r, "val_r2_median", "metrics.val_metrics.r2_median") is not None]
    if candidates:
        best = max(candidates, key=lambda r: _metric_float(r, "val_r2_median", "metrics.val_metrics.r2_median") or -999.0)
        return _full_record_brief(best)
    return {
        "exp_id": state.get("best_experiment_id") or state.get("best_exp_id"),
        "arch": state.get("best_arch_name") or state.get("best_arch"),
        "config": state.get("best_config"),
        "val_r2_median": state.get("best_r2_median"),
        "round": None,
        "source": "workflow_state_fallback",
    }


def build_all_time_top_k(history: list[dict], full_results: list[dict], k: int = 12) -> list[dict]:
    """Top-K scored full completed runs across all available history plus current round."""
    by_id: dict[str, dict] = {}
    for record in [*history, *full_results]:
        if not _is_full_completed(record):
            continue
        brief = _full_record_brief(record)
        if brief.get("val_r2_median") is None:
            continue
        key = str(brief.get("exp_id") or id(record))
        by_id[key] = brief
    return sorted(by_id.values(), key=lambda r: r.get("val_r2_median") or -999.0, reverse=True)[:k]


def _compact_knowledge_item(item, source: str, table: str) -> dict:
    if isinstance(item, dict):
        text = item.get("claim") or item.get("pattern") or item.get("advisory") or item.get("reason") or item.get("summary") or item.get("note")
        return {
            "source": source,
            "table": table,
            "id": item.get("id") or item.get("hypothesis_id"),
            "text": text,
            "outcome": item.get("outcome"),
            "confidence": item.get("confidence"),
            "evidence_run_ids": item.get("evidence_run_ids") or item.get("contradicting_evidence_run_ids") or item.get("evidence"),
            "next_test": item.get("next_test"),
        }
    return {"source": source, "table": table, "text": str(item)[:500]}


def build_hypothesis_status_table(state: dict, history: list[dict], max_items: int = 40) -> list[dict]:
    """Compact deterministic table from recent review/knowledge blocks."""
    sources: list[tuple[str, dict]] = []
    if isinstance(state.get("round_review"), dict):
        sources.append(("state.round_review", state["round_review"]))
    for idx, item in enumerate(history[-80:]):
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("round_review"), dict):
            sources.append((f"history[-80:][{idx}].round_review", item["round_review"]))
        elif isinstance(item.get("knowledge_update"), dict) or any(k in item for k in ("hypothesis_resolution_log", "recommended_hypotheses")):
            sources.append((f"history[-80:][{idx}]", item))
    campaign_dir = Path(os.environ.get("HYBRID_CAMPAIGN_DIR", PROJECT_ROOT / "campaigns" / "hybrid"))
    artifacts_dir = campaign_dir / "artifacts"
    if artifacts_dir.exists():
        review_paths = sorted(artifacts_dir.glob("r*/round_review.json"))[-5:]
        for path in review_paths:
            try:
                review = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(review, dict):
                sources.append((str(path.relative_to(campaign_dir)), review))

    rows: list[dict] = []
    keys = ("hypothesis_resolution_log", "recommended_hypotheses", "positive_patterns", "negative_patterns", "contradicted_patterns")
    for source, review in sources[-10:]:
        ku = review.get("knowledge_update") if isinstance(review.get("knowledge_update"), dict) else review
        for key in keys:
            value = ku.get(key) if isinstance(ku, dict) else None
            if isinstance(value, list):
                for item in value:
                    rows.append(_compact_knowledge_item(item, source, key))
            elif value:
                rows.append(_compact_knowledge_item(value, source, key))
    return rows[-max_items:]


def build_family_summary_recent(history: list[dict], full_results: list[dict], max_rounds: int = 5) -> list[dict]:
    """Very small family-level summary over recent full rounds."""
    recent_rounds = {r["round"] for r in build_recent_rounds_summary(history, full_results, max_rounds=max_rounds)}
    families: dict[str, dict] = {}
    for record in [*history, *full_results]:
        if _round_from_record(record) not in recent_rounds or not _is_full_completed(record):
            continue
        arch = _full_record_brief(record).get("arch") or "unknown"
        family = str(arch).replace("_unet", "").split("_")[0]
        fam = families.setdefault(family, {"family": family, "count": 0, "best_val_r2_median": None, "best_exp_id": None})
        fam["count"] += 1
        r2 = _metric_float(record, "val_r2_median", "metrics.val_metrics.r2_median")
        if r2 is not None and (fam["best_val_r2_median"] is None or r2 > fam["best_val_r2_median"]):
            fam["best_val_r2_median"] = r2
            fam["best_exp_id"] = record.get("exp_id") or record.get("experiment_id")
    return sorted(families.values(), key=lambda r: r.get("best_val_r2_median") or -999.0, reverse=True)[:20]


def build_full_review_context_bundle(state: dict, history: list[dict], full_results: list[dict]) -> dict:
    return {
        "recent_rounds_summary": build_recent_rounds_summary(history, full_results, max_rounds=5),
        "current_best_anchor": build_current_best_anchor(state, history, full_results),
        "all_time_top_k": build_all_time_top_k(history, full_results, k=12),
        "hypothesis_status_table": build_hypothesis_status_table(state, history, max_items=40),
        "family_summary_recent": build_family_summary_recent(history, full_results, max_rounds=5),
    }


def build_full_review_prompt(state: dict, history: list[dict], full_results: list[dict]) -> str:
    best_r2 = state.get("best_r2_median", -1)
    prev_review = state.get("round_review")
    round_num = state.get("round_num", 0)
    review_context_bundle = build_full_review_context_bundle(state, history, full_results)

    return (
        "You are a cross-round scientific auditor for Hybrid NAS. "
        "First audit the evidence tables and current-round full results; only after that give soft recommendations for the planner.\n"
        f"Round: {round_num}\n"
        f"Best R2_median ever from workflow_state: {best_r2:.4f}\n"
        f"Previous review: {json.dumps(prev_review, ensure_ascii=False) if prev_review else 'None'}\n\n"
        "review_context_bundle (deterministic pre-summary; use these table names in evidence_table_refs):\n"
        f"{json.dumps(review_context_bundle, indent=2, ensure_ascii=False)}\n\n"
        f"This round full_results ({len(full_results)} experiments; preserve as primary current-round evidence):\n"
        f"{json.dumps(full_results, indent=2, ensure_ascii=False)}\n\n"
        "Small raw history tail for provenance/debug only, not the main cross-round evidence source:\n"
        f"{json.dumps(history[-5:], indent=2, ensure_ascii=False)}\n\n"
        "Analyze/audit in this order:\n"
        "1. Evidence scope and table sanity: current full_results, recent_rounds_summary, current_best_anchor, all_time_top_k, hypothesis_status_table.\n"
        "2. Which architectures performed best/worst and why, citing exp_ids.\n"
        "3. R2 trends across rounds (improving, stagnating, declining), using recent_rounds_summary and all_time_top_k.\n"
        "4. Specific recommendations for next round, explicitly marked soft/hard with evidence and next_test.\n"
        "5. Whether to continue searching or stop (action: continue|done).\n"
        "6. Why the current Sequential benchmark/current-best remains strong, citing Sequential exp_ids.\n"
        "7. Which failures are resource, optimization, generalization, architecture-bias, code, or config/schema failures.\n"
        "8. Resolve prior hypotheses if possible: supported/refuted/inconclusive/continued using hypothesis_status_table.\n"
        "9. Mandatory cross-round audit: compare this round against recent best trend, historical best runs, near-config contrasts, family saturation, and smoke-vs-full bias. Distinguish evidence-backed conclusions from uncertain hypotheses.\n\n"
        "Review policy: audit conclusions are soft scientific evidence unless they identify a schema-invalid config, resource-infeasible config, locked train/data/eval contract violation, explicit user/controller rule violation, or another locked-contract breach. Your recommendations are SOFT evidence for the planner, not commands. Mark something hard only for those locked/resource/schema/controller cases. Do not hard-ban EMA, architecture families, input feature contracts, or loss choices solely because this round underperformed; instead record confidence, contradicting evidence, and ablation/diversity opportunities.\n"
        "Return JSON with the legacy fields plus a machine-readable knowledge_update block.\n"
        "Use only Sequential/Hybrid results in this prompt and provided history/context. Do not use or infer V3 performance.\n"
        "Failure_class enum: PASS, RESOURCE_OOM, RESOURCE_HELD, RESOURCE_EVICTED, CODEGEN_BUG, SCHEMA_MISMATCH, SHAPE_ERROR, OPTIMIZATION_UNSTABLE, OVERFIT, UNDERFIT, BENCHMARK_TIE, LOW_VALUE_VARIANT, AUTO_FAIL, UNKNOWN.\n\n"
        "Return JSON. Keep legacy fields, but prefer object-shaped recommendations when possible. String recommendations remain allowed for backward compatibility.\n"
        '{"action": "continue|done", "summary": "...", "top_performers": [{"exp_id": "...", "arch_name": "...", "val_r2_median": 0.0, "round": 0, "reason": "..."}], '
        '"bottom_performers": [...], "recommendations": [{"id": "R...", "recommendation": "...", "hardness": "soft|hard", "hardness_reason": "...", "confidence": "low|medium|high", "evidence_run_ids": [...], "evidence_table_refs": ["recent_rounds_summary", "all_time_top_k"], "next_test": "..."}], '
        '"stagnation_warning": true/false, "r2_trend": "improving|stagnating|declining", '
        '"cross_round_audit": {"evidence_scope": "...", "audit_findings": [{"id": "A...", "finding": "...", "table": "recent_rounds_summary|current_best_anchor|all_time_top_k|hypothesis_status_table|family_summary_recent|full_results", "evidence_run_ids": [...], "confidence": "low|medium|high"}], "recent_best_trend": "...", "historical_best_contrast": "...", "near_config_contrasts": [...], "contradicted_hypotheses": [...], "uncertain_hypotheses": [...], "family_saturation_signals": [...], "smoke_full_bias_notes": [...], "recommendation_confidence": {"overall": "low|medium|high", "by_recommendation": [...]}}, '
        '"knowledge_update": {"benchmark_survival_explanation": {"summary": "...", "cited_failed_attempts": [...]}, '
        '"failure_taxonomy": [{"id": "F...", "arch_name": "...", "failure_class": "...", "evidence_run_ids": [...], "planner_implication": "..."}], '
        '"positive_patterns": [{"id": "P...", "pattern": "...", "evidence_run_ids": [...], "planner_implication": "..."}], '
        '"negative_patterns": [...], "recommended_hypotheses": [{"id": "H...", "claim": "...", "source": "Sequential-result-driven", "evidence_so_far": [...], "next_test": "..."}], '
        '"hypothesis_resolution_log": [{"hypothesis_id": "H...", "outcome": "supported|refuted|inconclusive|continued", "evidence": [...], "reason": "..."}], '
        '"soft_advisories": [{"advisory": "...", "confidence": "low|medium|high", "evidence_run_ids": [...], "planner_use": "soft evidence only"}], '
        '"contradicted_patterns": [{"pattern": "...", "contradicting_evidence_run_ids": [...], "interpretation": "..."}], '
        '"search_policy_notes": [{"note": "...", "hardness": "soft|hard", "reason": "..."}], '
        '"cooldowns": [{"arch_family": "...", "reason": "...", "evidence_run_ids": [...], "rounds": 2, "auto_thaw_condition": "...", "hardness": "soft unless resource/schema/contract"}]}}\n'
    )

def call_claude_review(prompt: str, art_dir: Path) -> dict | None:
    """Call Claude Opus for full review."""
    cache = art_dir / "review_claude.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    ok, out = run_cmd(
        [CLAUDE_BIN_FULL, "--model", "opus", "--permission-mode",
         "bypassPermissions", "--print"],
        timeout=900, stdin_text=prompt,
    )
    if ok and out:
        result = extract_json_with_key(out, "action")
        if result:
            cache.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result
    return None


def call_codex_review(prompt: str, art_dir: Path) -> dict | None:
    """Call Codex for full review."""
    cache = art_dir / "review_codex.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    ok, out = run_cmd(
        [CODEX_BIN_FULL, "exec", "--model", CODEX_MODEL, "--skip-git-repo-check",
         "--cd", "/tmp", "--ephemeral", "--ignore-rules"],
        timeout=900, stdin_text=prompt,
    )
    if ok and out:
        result = extract_json_with_key(out, "action")
        if result:
            cache.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result
    return None


def _normalize_recommendation(item, source: str) -> dict | str:
    """Prefer object recommendations while preserving old string compatibility."""
    if isinstance(item, dict):
        out = dict(item)
        out.setdefault("source", source)
        out.setdefault("hardness", "soft")
        out.setdefault("confidence", out.get("confidence") or "unknown")
        out.setdefault("evidence_run_ids", out.get("evidence_run_ids") or [])
        out.setdefault("evidence_table_refs", out.get("evidence_table_refs") or [])
        if "recommendation" not in out:
            out["recommendation"] = out.get("advisory") or out.get("note") or out.get("claim") or str(item)[:500]
        return out
    return item


def _normalize_top_performer(item, source: str) -> dict | str:
    """Normalize object top performers without breaking legacy strings."""
    if isinstance(item, dict):
        out = dict(item)
        out.setdefault("source", source)
        if "exp_id" not in out and "experiment_id" in out:
            out["exp_id"] = out.get("experiment_id")
        if "val_r2_median" not in out:
            r2 = _metric_float(out, "val_r2_median", "metrics.val_metrics.r2_median")
            if r2 is not None:
                out["val_r2_median"] = r2
        return out
    return item



def _top_performer_key(item: dict | str) -> tuple:
    if not isinstance(item, dict):
        return ("string", str(item)[:160])
    eid = str(item.get("exp_id") or item.get("experiment_id") or "")
    if eid:
        return ("exp_id", eid)
    return ("semantic", _semantic_signature_from_brief(item))


def _dedupe_top_performers(items: list[dict | str]) -> tuple[list[dict | str], list[dict]]:
    groups: dict[tuple, list[dict | str]] = {}
    for item in items:
        groups.setdefault(_top_performer_key(item), []).append(item)
    kept: list[dict | str] = []
    duplicate_groups: list[dict] = []
    def score(x):
        if isinstance(x, dict):
            v = x.get("val_r2_median")
            return float(v) if isinstance(v, (int, float)) else -999.0
        return -999.0
    for key, group in groups.items():
        best = sorted(group, key=score, reverse=True)[0]
        kept.append(best)
        if len(group) > 1:
            duplicate_groups.append({
                "kind": key[0],
                "key": list(key[1]) if isinstance(key[1], tuple) else key[1],
                "kept": best.get("exp_id") if isinstance(best, dict) else str(best)[:80],
                "members": [g.get("exp_id") if isinstance(g, dict) else str(g)[:80] for g in group],
            })
    kept.sort(key=score, reverse=True)
    return kept, duplicate_groups


_SEED_ACTION_RE = re.compile(
    r"\b(multi[- ]?seed|seed replication|seed_count|seed count|seed[- ]?check|seeds?\s*[=:]?\s*[\[{]|replicate\s+seeds?|repeat\s+seeds?|multiple\s+seeds?)\b",
    re.IGNORECASE,
)


def _contains_forbidden_seed_action(item) -> bool:
    try:
        text = json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
    except TypeError:
        text = str(item)
    return bool(_SEED_ACTION_RE.search(text))


def _looks_hard(item) -> bool:
    return isinstance(item, dict) and str(item.get("hardness") or "").lower() == "hard"


def _has_evidence_ids(item) -> bool:
    vals = item.get("evidence_run_ids") if isinstance(item, dict) else None
    return isinstance(vals, list) and bool(vals)


def _is_resource_hard_rule(item) -> bool:
    if not _looks_hard(item):
        return True
    text = json.dumps(item, ensure_ascii=False).lower()
    return any(tok in text for tok in ("resource", "oom", "vram", "memory", "schema", "contract", "invalid", "controller"))


def _split_planner_layers(review: dict) -> None:
    blocked: list[dict] = []
    actionable: list[dict | str] = []
    soft_evidence: list[dict | str] = []
    tooling_defects: list[dict] = []

    for rec in review.get("recommendations", []) or []:
        if _contains_forbidden_seed_action(rec):
            b = dict(rec) if isinstance(rec, dict) else {"recommendation": str(rec)}
            b["blocked_by_user_seed_policy"] = True
            b["block_reason"] = "the human researcher hard rule: do not propose seed replication/multi-seed/seed_count gates as planner action."
            blocked.append(b)
            continue
        if isinstance(rec, dict) and str(rec.get("hardness") or "soft").lower() == "hard":
            actionable.append(rec)
        else:
            soft_evidence.append(rec)

    ku = review.get("knowledge_update") if isinstance(review.get("knowledge_update"), dict) else {}
    notes = ku.get("search_policy_notes") if isinstance(ku.get("search_policy_notes"), list) else []
    filtered_notes = []
    for note in notes:
        if _contains_forbidden_seed_action(note):
            b = dict(note) if isinstance(note, dict) else {"note": str(note)}
            b["blocked_by_user_seed_policy"] = True
            b["source_section"] = "knowledge_update.search_policy_notes"
            blocked.append(b)
        else:
            filtered_notes.append(note)
    if ku:
        ku["search_policy_notes"] = filtered_notes

    for key in ("recommended_hypotheses", "soft_advisories", "positive_patterns", "negative_patterns"):
        vals = ku.get(key) if isinstance(ku.get(key), list) else []
        kept_vals = []
        for item in vals:
            if _contains_forbidden_seed_action(item):
                b = dict(item) if isinstance(item, dict) else {"item": str(item)}
                b["source_section"] = f"knowledge_update.{key}"
                b["blocked_by_user_seed_policy"] = True
                blocked.append(b)
            else:
                kept_vals.append(item)
                if key in ("soft_advisories", "recommended_hypotheses"):
                    soft_evidence.append({"source_section": f"knowledge_update.{key}", "item": item})
        if ku and key in ku:
            ku[key] = kept_vals

    cra = review.get("cross_round_audit") if isinstance(review.get("cross_round_audit"), dict) else {}
    for finding in cra.get("audit_findings", []) if isinstance(cra.get("audit_findings"), list) else []:
        if isinstance(finding, dict) and any(tok in json.dumps(finding, ensure_ascii=False).lower() for tok in ("overcount", "stale", "duplicate", "tool", "summary")):
            tooling_defects.append(finding)

    review["planner_actionable"] = actionable
    review["soft_evidence"] = soft_evidence
    review["blocked_recommendations"] = blocked
    review["tooling_defects"] = tooling_defects
    if ku:
        ku.setdefault("planner_actionable", actionable)
        ku.setdefault("soft_evidence", soft_evidence)
        ku.setdefault("blocked_recommendations", blocked)
        ku.setdefault("tooling_defects", tooling_defects)


def build_review_quality_report(review: dict, full_results: list[dict], full_results_path: Path) -> dict:
    source_count = len(full_results)
    completed_count = sum(1 for r in full_results if str(r.get("status") or "").lower() == "completed")
    failed_count = sum(1 for r in full_results if str(r.get("status") or "").lower() == "failed")
    m = re.search(r"r(\d{3})", str(full_results_path.parent.name))
    current_round = int(m.group(1)) if m else None

    recent = build_recent_rounds_summary([], full_results, max_rounds=5)
    current = next((x for x in recent if x.get("round") == current_round), recent[-1] if recent else {})
    summary_matches = (
        current.get("source_record_count") == source_count and
        current.get("completed_count") == completed_count and
        current.get("failed_count") == failed_count
    )
    review["recent_rounds_summary"] = recent

    top_deduped, dup_groups = _dedupe_top_performers(list(review.get("top_performers") or []))
    review["top_performers"] = top_deduped
    if dup_groups:
        review["top_performer_duplicate_groups"] = dup_groups
    top_duplicate_free = len(dup_groups) == 0

    planner_text = json.dumps({
        "planner_actionable": review.get("planner_actionable", []),
        "search_policy_notes": (review.get("knowledge_update") or {}).get("search_policy_notes", []) if isinstance(review.get("knowledge_update"), dict) else [],
    }, ensure_ascii=False)
    forbidden_seed_in_planner = bool(_SEED_ACTION_RE.search(planner_text))
    hard_recs = [r for r in review.get("planner_actionable", []) if _looks_hard(r)]
    hard_have_evidence = all(_has_evidence_ids(r) for r in hard_recs)
    hard_resource_ok = all(_is_resource_hard_rule(r) for r in hard_recs)
    checks = {
        "current_round_source_count_matches_full_results": summary_matches,
        "top_performers_duplicate_free": top_duplicate_free,
        "no_forbidden_seed_action_in_planner_fields": not forbidden_seed_in_planner,
        "hard_recommendations_have_evidence_run_ids": hard_have_evidence,
        "resource_hard_rules_limited_to_resource_oom_schema_contract": hard_resource_ok,
    }
    return {
        "ok": all(checks.values()),
        "timestamp": now_iso(),
        "source_file": str(full_results_path),
        "source_record_count": source_count,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "current_round_summary": current,
        "top_duplicate_groups": dup_groups,
        "checks": checks,
        "notes": "Quality report is advisory and does not block workflow.",
    }


def synthesize_full_review(claude_result: dict | None, codex_result: dict | None,
                           full_results: list[dict]) -> dict:
    """Merge dual-model full reviews with deterministic dedupe and planner-safe layers."""
    review = {"action": "continue", "summary": "", "top_performers": [],
              "recommendations": [], "stagnation_warning": False,
              "stagnation_signals": []}

    claude_action = (claude_result or {}).get("action")
    codex_action = (codex_result or {}).get("action")
    if claude_action == "done" or codex_action == "done":
        review["action"] = "done"
        reason_src = claude_result if claude_action == "done" else codex_result
        review["summary"] = (reason_src or {}).get("summary", "Dual review: done recommended")
    else:
        review["action"] = "continue"

    summaries = []
    for source_name, src in (("claude", claude_result), ("codex", codex_result)):
        if isinstance(src, dict) and src.get("summary"):
            summaries.append(f"{source_name}: {src.get('summary')}")
    if summaries and (review["action"] == "continue" or not review.get("summary")):
        review["summary"] = " | ".join(summaries)

    seen_recs = set()
    for source_name, src in (("claude", claude_result), ("codex", codex_result)):
        if not src:
            continue
        for rec in src.get("recommendations", []):
            rec_norm = _normalize_recommendation(rec, source_name)
            rec_key = str(rec_norm)[:160]
            if rec_key not in seen_recs:
                seen_recs.add(rec_key)
                review["recommendations"].append(rec_norm)
        if src.get("top_performers"):
            review["top_performers"].extend(_normalize_top_performer(tp, source_name) for tp in src["top_performers"])
    review["top_performers"], review["top_performer_duplicate_groups"] = _dedupe_top_performers(review["top_performers"])

    claude_stag = (claude_result or {}).get("stagnation_warning", False)
    codex_stag = (codex_result or {}).get("stagnation_warning", False)
    review["stagnation_warning"] = claude_stag and codex_stag
    for source_name, src in (("claude", claude_result), ("codex", codex_result)):
        if isinstance(src, dict) and src.get("stagnation_warning"):
            review["stagnation_signals"].append({"source": source_name, "stagnation_warning": True, "r2_trend": src.get("r2_trend"), "summary": src.get("summary")})

    cross_round_audit = {"raw": {}, "recommendation_confidence": [], "audit_findings": []}
    for source_name, src in (("claude", claude_result), ("codex", codex_result)):
        if not isinstance(src, dict):
            continue
        audit = src.get("cross_round_audit")
        if isinstance(audit, dict):
            cross_round_audit["raw"][source_name] = audit
            conf = audit.get("recommendation_confidence")
            if conf is not None:
                cross_round_audit["recommendation_confidence"].append({"source": source_name, "recommendation_confidence": conf})
            findings = audit.get("audit_findings")
            if isinstance(findings, list):
                for finding in findings:
                    if isinstance(finding, dict):
                        preserved = dict(finding); preserved.setdefault("source", source_name)
                        cross_round_audit["audit_findings"].append(preserved)
                    else:
                        cross_round_audit["audit_findings"].append({"source": source_name, "finding": finding})
            for key in ("evidence_scope", "recent_best_trend", "historical_best_contrast", "near_config_contrasts", "contradicted_hypotheses", "uncertain_hypotheses", "family_saturation_signals", "smoke_full_bias_notes"):
                value = audit.get(key)
                if value:
                    if isinstance(value, list):
                        cross_round_audit.setdefault(key, []).extend(value)
                    else:
                        cross_round_audit.setdefault(key, []).append({"source": source_name, "value": value})
    if cross_round_audit["raw"]:
        review["cross_round_audit"] = cross_round_audit

    knowledge_update = {}
    for src in [claude_result, codex_result]:
        if not isinstance(src, dict):
            continue
        ku = src.get("knowledge_update")
        if isinstance(ku, dict):
            for key, value in ku.items():
                if isinstance(value, list):
                    knowledge_update.setdefault(key, []).extend(value)
                elif key not in knowledge_update or not knowledge_update.get(key):
                    knowledge_update[key] = value
    if knowledge_update:
        review["knowledge_update"] = knowledge_update

    _split_planner_layers(review)
    review["claude"] = claude_result
    review["codex"] = codex_result
    return review

def extract_json_with_key(text: str, required_key: str) -> dict | None:
    """Extract JSON object containing required_key from AI output."""
    # Try fenced code block first
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict) and required_key in data:
                return data
        except json.JSONDecodeError:
            pass
    # Use json.JSONDecoder for proper nested brace handling
    decoder = json.JSONDecoder()
    candidates = []
    pos = 0
    while pos < len(text):
        idx = text.find('{', pos)
        if idx == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                candidates.append(obj)
            pos = end
        except json.JSONDecodeError:
            pos = idx + 1
    # Prefer candidate with required_key
    for c in candidates:
        if required_key in c:
            return c
    return candidates[-1] if candidates else None


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> None:
    campaign_dir = Path(os.environ.get("HYBRID_CAMPAIGN_DIR", "."))
    state = load_state(campaign_dir)
    round_num = state.get("round_num", 0)
    art_dir = round_artifact_dir(campaign_dir, round_num)
    phase = state.get("phase")

    if phase == "post_codegen_review":
        manifest_path = art_dir / "codegen_manifest.json"
        if not manifest_path.exists():
            print(json.dumps({"ok": False, "reason": "missing codegen_manifest.json"}, ensure_ascii=False))
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # All generated_archs must be reviewed.  Registered/shared architectures
        # are no longer allowed to be considered validated merely because they
        # exist in the shared registry; codegen must create an isolated generated
        # model for each proposal and post-codegen review must validate it.
        target_archs = list(manifest.get("generated_archs", []))
        if not target_archs:
            manifest_failures = _manifest_blocking_failures(manifest)
            result = {
                "ok": len(manifest_failures) == 0,
                "reviewed_archs": [],
                "validated_archs": manifest.get("validated_archs", []),
                "failed": manifest_failures,
                "manifest_failures": manifest_failures,
                "timestamp": now_iso(),
            }
            (art_dir / "post_codegen_review.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({"ok": result["ok"], "action": "no_generated_targets"}, ensure_ascii=False))
            return

        prompt = build_post_codegen_prompt(state, manifest, target_archs)
        sources = {"claude_initial": False, "codex": False, "claude_final": False}

        # Tests can set this to exercise the branch without spending AI calls.
        if os.environ.get("V4_POST_CODEGEN_REVIEW_SKIP_AI") != "1":
            ok1, out1 = call_claude_post_codegen(prompt, art_dir, "initial")
            sources["claude_initial"] = ok1
            codex_prompt = prompt + "\n\nClaude initial review output:\n" + (out1 or "")[-4000:]
            ok2, out2 = call_codex_post_codegen(codex_prompt, art_dir)
            sources["codex"] = ok2
            final_prompt = prompt + "\n\nClaude initial output:\n" + (out1 or "")[-2500:] + "\n\nCodex review output:\n" + (out2 or "")[-2500:]
            ok3, _ = call_claude_post_codegen(final_prompt, art_dir, "final")
            sources["claude_final"] = ok3

        result = run_post_codegen_validation(state, manifest, target_archs)
        result["sources"] = sources
        result["timestamp"] = now_iso()
        (art_dir / "post_codegen_review.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"ok": result["ok"], "validated": len(result["validated_archs"]),
                          "failed": len(result["failed"]), "sources": sources}, ensure_ascii=False))

    elif phase == "smoke_classify":
        # â”€â”€ Failure diagnosis (Sonnet + Codex) â”€â”€
        # Despite the phase name, controller REPAIR decisions can come from
        # either smoke or full collection.  Use the controller tag to choose
        # the matching result artifact, otherwise full-run repair failures are
        # invisible here and the workflow incorrectly proceeds to another full
        # submit.
        decision_path = art_dir / "controller_decision.json"
        decision = {}
        if decision_path.exists():
            try:
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                decision = {}
        result_tag = decision.get("tag") or state.get("last_collect_tag") or state.get("submit_tag") or "smoke"
        results_key = "full_results" if result_tag == "full" else "smoke_results"
        result_file = art_dir / ("full_results.json" if result_tag == "full" else "smoke_results.json")

        diagnosis_results = state.get(results_key, [])
        if not diagnosis_results and result_file.exists():
            diagnosis_results = json.loads(result_file.read_text(encoding="utf-8"))

        failures = [r for r in diagnosis_results if r["status"] != "completed"]

        # If the deterministic controller already decided which failures are
        # true REPAIR items, diagnose only those.  OOM / scheduler / transient
        # RETRY items must not be fed into codegen as "fixable config" tasks,
        # otherwise repair tries to rewrite model code for a resource problem.
        repair_ids = {
            row.get("run_id") for row in decision.get("per_run", [])
            if row.get("action") == "REPAIR" and row.get("run_id")
        }
        if repair_ids:
            failures = [
                r for r in failures
                if (r.get("experiment_id") or r.get("exp_id")) in repair_ids
            ]
        else:
            failures = []

        if not failures:
            (art_dir / "smoke_fix_plan.json").write_text(
                json.dumps({"has_fixable": False, "source_tag": result_tag}, ensure_ascii=False), encoding="utf-8")
            print(json.dumps({"ok": True, "action": "no_repair_failures", "source_tag": result_tag}, ensure_ascii=False))
            return

        prompt = build_smoke_diagnosis_prompt(failures)

        # Dual model call
        claude_result = call_claude_smoke(prompt, art_dir)
        codex_result = call_codex_smoke(prompt, art_dir)

        fix_plan = synthesize_smoke_diagnosis(claude_result, codex_result)

        # Fallback: if both failed, use rule-based diagnosis
        if not claude_result and not codex_result:
            fix_plan = rule_based_smoke_diagnosis(failures)

        fix_plan = attach_config_patches(fix_plan, diagnosis_results)
        fix_plan["source_tag"] = result_tag

        (art_dir / "smoke_fix_plan.json").write_text(
            json.dumps(fix_plan, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"ok": True, "fixable": fix_plan.get("has_fixable", False),
                          "fixes": len(fix_plan.get("fixes", [])),
                          "sources": {"claude": claude_result is not None,
                                      "codex": codex_result is not None}},
                         ensure_ascii=False))

    elif phase == "review":
        # â”€â”€ Full review (Opus + Codex) â”€â”€
        full_results_path = art_dir / "full_results.json"
        full_results = []
        if full_results_path.exists():
            full_results = json.loads(full_results_path.read_text(encoding="utf-8"))

        history = []
        hpath = history_path(campaign_dir)
        if hpath.exists():
            for line in hpath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        prompt = build_full_review_prompt(state, history, full_results)

        # Dual model call
        claude_result = call_claude_review(prompt, art_dir)
        codex_result = call_codex_review(prompt, art_dir)

        review = synthesize_full_review(claude_result, codex_result, full_results)
        quality_report = build_review_quality_report(review, full_results, full_results_path)
        review["review_quality_report_path"] = str(art_dir / "review_quality_report.json")

        (art_dir / "review_quality_report.json").write_text(
            json.dumps(quality_report, indent=2, ensure_ascii=False), encoding="utf-8")
        (art_dir / "round_review.json").write_text(
            json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            knowledge = build_experiment_knowledge(campaign_dir, state, round_num, review)
            review["experiment_knowledge_path"] = str(art_dir / "experiment_knowledge.json")
            (art_dir / "round_review.json").write_text(
                json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            knowledge = None
            (art_dir / "experiment_knowledge_error.txt").write_text(str(exc), encoding="utf-8")
        print(json.dumps({"ok": True, "action": review.get("action"),
                          "knowledge_written": knowledge is not None,
                          "sources": {"claude": claude_result is not None,
                                      "codex": codex_result is not None}},
                         ensure_ascii=False))

    else:
        print(json.dumps({"ok": False, "reason": f"reviewer idle on phase={phase}"}, ensure_ascii=False))


def rule_based_smoke_diagnosis(failures: list[dict]) -> dict:
    """Fallback when both AI calls fail: simple pattern-based diagnosis."""
    fixes = []
    for f in failures:
        log = f.get("log_tail", "")
        status = f.get("status", "")
        diagnosis = "unknown"
        fixable = False
        fix_description = ""
        fix_type = "code"

        if status == "crashed" or "Traceback" in log:
            if "CUDA out of memory" in log or "OOM" in log:
                diagnosis = "OOM"
                fixable = True
                fix_description = "Reduce n_c or depth to lower VRAM usage"
                fix_type = "config"
            elif "shape" in log.lower() or "size mismatch" in log.lower():
                diagnosis = "shape_mismatch"
                fixable = True
                fix_description = "Fix tensor dimension mismatch in forward pass"
                fix_type = "code"
            elif "Import" in log or "ModuleNotFound" in log:
                diagnosis = "import_error"
                fixable = True
                fix_description = "Fix import path or add missing dependency"
                fix_type = "code"
            else:
                diagnosis = "runtime_crash"
                fixable = True
                fix_description = "Investigate traceback and fix the error"
                fix_type = "code"
        elif status == "loss_nan":
            diagnosis = "loss_nan"
            fixable = True
            fix_description = "Add gradient clipping or reduce learning rate"
            fix_type = "config"
        elif status == "submit_failed":
            diagnosis = "submit_failure"
            fixable = False
            fix_description = "Job submission failed, likely infrastructure issue"

        fixes.append({
            "exp_id": f.get("exp_id", ""),
            "arch_name": f.get("arch_name", ""),
            "diagnosis": diagnosis,
            "fixable": fixable,
            "fix_description": fix_description,
            "fix_type": fix_type,
        })

    return {
        "has_fixable": any(f["fixable"] for f in fixes),
        "fixes": fixes,
        "source": "rule_based_fallback",
    }


if __name__ == "__main__":
    main()



