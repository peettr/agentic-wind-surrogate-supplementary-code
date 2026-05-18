"""Sequential Campaign Runner â€” State machine with smoke test loop.

Phases:
  propose â†’ codegen â†’ submit(20ep) â†’ monitor(30min) â†’ collect â†’ smoke_classify
    â†’ (ai_fix â†’ codegen loop, max 3) or
    â†’ submit(200ep) â†’ monitor(180min) â†’ collect â†’ review â†’ next_round

Runner calls:
  - Executor directly for SSH/Condor operations (submit, monitor, collect)
  - Worker subprocesses for AI operations (planner, codegen, reviewer)

Triggered by Task Scheduler every 5min. Runner runs once, advances state, exits.

Usage:
    python workflow_runner.py --campaign-dir campaigns/baseline
    python workflow_runner.py --campaign-dir campaigns/baseline --resume
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "workflow_engine"))
sys.path.insert(0, str(PROJECT_ROOT / "explorer"))

from workflow_common import (
    WORKSPACE, now_iso, load_state, save_state, fresh_state,
    round_artifact_dir, experiment_id, history_path,
    SMOKE_EPOCHS, SMOKE_MAX_FIX_ROUNDS, FULL_EPOCHS,
    EXPERIMENTS_PER_ROUND, resource_feasibility_guard,
)
from workflow_executor import Executor, JobHandle
from attempt_manifest import load_manifest, save_manifest, record_attempt, ensure_run, check_limit, base_run_id, TERMINAL_STATUSES
from failure_classifier import classify_result
from schema_guards import validate_experiment_schema

# Generated/reference model directories. Runtime jobs must use per-run model
# copies. shared/models is allowed only as a reference source, never as the
# runtime import target.
GENERATED_DIR = Path(__file__).resolve().parent.parent / "models" / "generated"
SHARED_MODELS_DIR = Path(__file__).resolve().parent.parent / "shared" / "models"

LOG_PATH = WORKSPACE / "hybrid" / "workflow_runner.log"
LOCK_PATH = WORKSPACE / "hybrid" / "workflow_runner.lock"
LOG_MAX_BYTES = 512 * 1024
H100_RETRY_WAIT_TIMEOUT_MIN = 180
MONITOR_RETRY_MAX_PER_TICK = 2
MEMORY_RETRY_CLASSES = {"CUDA_OOM", "HIGH_VRAM", "CONDOR_MEMORY_LIMIT", "CONDOR_EVICTED_GPU_DOWNGRADE"}
LOCKED_WORKFLOW_FILES = [
    PROJECT_ROOT / "shared" / "train.py",
    PROJECT_ROOT / "shared" / "losses.py",
    PROJECT_ROOT / "shared" / "eval_module.py",
    PROJECT_ROOT / "templates" / "condor_submit.template",
    PROJECT_ROOT / "templates" / "condor_wrapper.sh",
]

# â”€â”€ Phase definitions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TERMINAL_PHASES = {"done", "failed", "blocked"}

VALID_TRANSITIONS: dict[str, set[str]] = {
    "propose":          {"codegen", "done", "blocked"},
    "codegen":          {"post_codegen_review", "submit", "ai_fix", "blocked"},
    "post_codegen_review": {"submit", "ai_fix", "blocked"},
    "submit":           {"monitor", "blocked", "failed"},
    "monitor":          {"collect", "blocked", "failed"},
    "collect":          {"controller", "failed", "blocked"},
    "controller":       {"monitor", "submit", "smoke_classify", "review", "blocked"},
    "smoke_classify":   {"ai_fix", "submit", "blocked"},
    "ai_fix":           {"codegen", "blocked"},
    "review":           {"next_round", "done", "blocked"},
    "next_round":       {"propose"},
    "done":             {"next_round"},
    "failed":           set(),
    "blocked":          set(),
}

WAIT_PHASES = {"monitor"}
MODEL_PHASES = {"review", "ai_fix", "codegen", "post_codegen_review", "propose", "smoke_classify", "controller"}


# â”€â”€ Logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def append_log(msg: str) -> None:
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > LOG_MAX_BYTES:
            old = LOG_PATH.with_suffix(".log.old")
            old.unlink(missing_ok=True)
            LOG_PATH.rename(old)
    except OSError:
        pass
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{now_iso()}] {msg}\n")


# â”€â”€ Lock â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def acquire_lock() -> bool:
    if LOCK_PATH.exists():
        try:
            lock_data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            lock_pid = int(lock_data.get("pid") or 0)
            if _pid_alive(lock_pid):
                return False
            lock_time = datetime.fromisoformat(lock_data.get("locked_at", ""))
            if lock_time.tzinfo is None:
                lock_time = lock_time.replace(tzinfo=timezone.utc)
            age_min = (datetime.now(timezone.utc) - lock_time).total_seconds() / 60.0
            if age_min < 240:
                return False
            append_log(f"Stale lock ({age_min:.1f} min old), removing")
        except (json.JSONDecodeError, ValueError, OSError):
            append_log("Corrupt lock, removing")
        LOCK_PATH.unlink(missing_ok=True)
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({"locked_at": now_iso(), "pid": os.getpid()}).encode("utf-8"))
        os.close(fd)
    except FileExistsError:
        return False
    return True


def release_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except OSError:
        pass


# â”€â”€ Worker invocation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _snapshot_locked_files() -> dict[Path, bytes]:
    snap: dict[Path, bytes] = {}
    for path in LOCKED_WORKFLOW_FILES:
        if path.exists():
            snap[path] = path.read_bytes()
    return snap


def _locked_file_hashes(snap: dict[Path, bytes]) -> dict[str, str]:
    return {str(path): hashlib.sha256(data).hexdigest() for path, data in snap.items()}


def run_worker(script_name: str, campaign_dir: Path) -> tuple[bool, str]:
    """Run a workflow worker script with locked-file protection."""
    script_path = Path(__file__).parent / script_name
    env = os.environ.copy()
    env["HYBRID_CAMPAIGN_DIR"] = str(campaign_dir)
    if script_name == "workflow_planner.py":
        env.setdefault("hybrid_WEB_QUERY_LIMIT", "6")
        env.setdefault("hybrid_WEB_QUERY_TIMEOUT_S", "120")
        env.setdefault("hybrid_WEB_REUSE_SOURCES", "1")
    before = _snapshot_locked_files()
    before_hashes = _locked_file_hashes(before)
    try:
        try:
            worker_timeout = int(env.get("hybrid_WORKER_TIMEOUT_S") or (5400 if script_name == "workflow_planner.py" else 1800))
        except (TypeError, ValueError):
            worker_timeout = 5400 if script_name == "workflow_planner.py" else 1800
        worker_python = env.get("hybrid_WORKER_PYTHON") or sys.executable
        r = subprocess.run(
            [worker_python, str(script_path)],
            capture_output=True, text=True, encoding="utf-8",
            env=env, timeout=worker_timeout,
        )
        text = ((r.stdout or "") + ("\n" + r.stderr if r.stderr else "")).strip()
    except subprocess.TimeoutExpired:
        return False, "worker timeout"

    after = _snapshot_locked_files()
    after_hashes = _locked_file_hashes(after)
    changed = [p for p, h in before_hashes.items() if after_hashes.get(p) != h]
    if changed:
        # Restore immediately.  AI workers may read locked files, but autonomous
        # repair/codegen/review must not modify training/eval/data infrastructure.
        for path, data in before.items():
            path.write_bytes(data)
        return False, f"locked files modified and restored: {changed}\n{text}"
    return r.returncode == 0, text


def get_executor(campaign_dir: Path) -> Executor:
    return Executor(campaign_dir)


def _apply_safe_config_patch(cfg: dict, patch: dict) -> dict:
    """Clone a proposal and apply a bounded config-repair patch.

    Only capacity/training knobs are accepted.  Data, seed, split, and feature
    settings are deliberately immutable in config repair.
    """
    allowed = {"n_c", "depth", "batch_size", "lr"}
    out = dict(cfg)
    applied = {k: v for k, v in (patch or {}).items() if k in allowed}
    for k, v in applied.items():
        out[k] = v
    out["_config_repair_patch"] = applied
    out["_repair_base_id"] = experiment_id(cfg)
    return out


def _apply_schema_config_patch(cfg: dict, patch: dict) -> dict:
    """Clone a proposal and attach a constructor-schema repair patch."""
    out = dict(cfg)
    out["_schema_repair_patch"] = {
        "arch_kwargs_remove": [str(k) for k in patch.get("arch_kwargs_remove", [])],
        "arch_kwargs_set": dict(patch.get("arch_kwargs_set", {}) or {}),
    }
    out["_repair_base_id"] = cfg.get("_repair_base_id") or experiment_id(cfg)
    return out


def _model_source_kind(path: Path) -> str:
    try:
        p = path.resolve()
        if GENERATED_DIR.resolve() in p.parents or p.parent == GENERATED_DIR.resolve():
            return "generated"
        if SHARED_MODELS_DIR.resolve() in p.parents or p.parent == SHARED_MODELS_DIR.resolve():
            return "reference_copy"
    except OSError:
        pass
    return "explicit"


def _resolve_model_source_for_attempt(
    arch_name: str,
    cfg: dict,
    art_dir: Path,
    state: dict,
    base_id: str,
) -> Path | None:
    """Resolve the local model source to copy into the run directory.

    shared/models is a reference source only. The executor always uploads the
    resolved file as an attempt-local model.py and points TrainConfig.script_path
    at that run-local copy.
    """
    raw_script = cfg.get("script_path")
    candidates: list[Path] = []
    if raw_script:
        raw = Path(str(raw_script))
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend([GENERATED_DIR / raw.name, SHARED_MODELS_DIR / raw.name, PROJECT_ROOT / raw])

    manifest_path = art_dir / "codegen_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            current_smoke_run_id = f"r{state['round_num']:03d}_smoke_{base_id}"
            validated_run_ids = set(manifest.get("validated_run_ids", []))
            validated_archs = set(manifest.get("validated_archs", []))
            skipped = set(manifest.get("skipped_registry_archs", []))
            if current_smoke_run_id in validated_run_ids or (arch_name in validated_archs and arch_name not in skipped):
                candidates.append(GENERATED_DIR / f"{arch_name}.py")
        except json.JSONDecodeError:
            pass

    # Final fallback is allowed only as a reference copy, not as a runtime import.
    candidates.append(SHARED_MODELS_DIR / f"{arch_name}.py")

    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _mark_fix_plan_unrepairable(camp: Path, state: dict, reason: str) -> int:
    """Terminal-fail current fix-plan targets that produced no runnable work."""
    art_dir = round_artifact_dir(camp, state["round_num"])
    fix_plan_path = art_dir / "smoke_fix_plan.json"
    if not fix_plan_path.exists():
        return 0
    try:
        fix_plan = json.loads(fix_plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    manifest = load_manifest(camp)
    changed = 0
    for fix in fix_plan.get("fixes", []):
        fid = fix.get("exp_id", "")
        if not fid:
            continue
        base_id = base_run_id(fid)
        entry = manifest.setdefault("runs", {}).get(base_id)
        if not entry:
            entry = manifest["runs"][base_id] = {
                "base_run_id": base_id,
                "run_type": "smoke",
                "max_retries": 3,
                "max_repairs": SMOKE_MAX_FIX_ROUNDS,
                "max_total_attempts": 5,
                "retry_count": 0,
                "repair_count": 0,
                "total_attempts": 0,
                "status": "ACTIVE",
                "config": {},
                "attempts": [],
                "repairs": [],
            }
        if entry.get("status") not in TERMINAL_STATUSES:
            entry["status"] = "AUTO_FAIL_UNREPAIRABLE"
            entry["terminal_reason"] = reason
            entry.setdefault("repairs", []).append({
                "run_id": fid,
                "status": "auto_failed_unrepairable",
                "reason": reason,
            })
            changed += 1
    if changed:
        save_manifest(camp, manifest)
    return changed


# â”€â”€ Generic handlers (executor calls, no AI) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def handle_submit(state: dict, camp: Path, epochs: int, prefix: str) -> dict:
    """Submit batch of experiments via Condor (GPU training)."""
    tag = "smoke" if epochs == SMOKE_EPOCHS else "full"
    append_log(f"Phase: SUBMIT ({tag}, {epochs}ep)")

    exc = get_executor(camp)
    ok, ssh_msg = exc.ssh_ok()
    if not ok:
        state["phase"] = "blocked"
        state["status_note"] = f"SSH unavailable: {ssh_msg}"
        return state

    proposals = state.get("proposals", [])

    # Controller-created retry submit: only resubmit selected semantic configs.
    retry_mode = bool(state.get("retry_mode")) and tag == state.get("retry_tag")
    retry_archs = set(state.get("retry_archs", []))
    retry_run_ids = set(state.get("retry_run_ids", []))
    retry_reasons = state.get("retry_reasons", {}) or {}
    high_vram_retry = retry_mode and any(
        str(v).upper() in MEMORY_RETRY_CLASSES
        for v in retry_reasons.values()
    )
    retry_gpu_requirements = None
    retry_request_memory_gb = 16
    if high_vram_retry:
        # Retrying memory failures on the same A40/L40S-sized resource is not a
        # meaningful retry. Escalate resource requirements while keeping
        # experiment config and data settings fixed.
        retry_gpu_requirements = 'regexp("qa-h100-", Machine) || regexp("qa-a100-", Machine)'
        # CUDA OOM is GPU VRAM pressure, not host RAM pressure. Keep host
        # RequestMemory at the normal 16GB default to avoid over-constraining
        # H100/A100 matching. Only true Condor cgroup memory-limit failures
        # get a modest host-RAM bump.
        retry_request_memory_gb = 32 if any(str(v).upper() == "CONDOR_MEMORY_LIMIT" for v in retry_reasons.values()) else 16
        append_log(f"retry-mode: memory failure detected, escalating to H100/A100 resources and request_memory={retry_request_memory_gb}GB")
    retry_manifest = load_manifest(camp) if retry_mode else {"runs": {}}

    if retry_mode:
        if retry_run_ids:
            def _retry_match(p: dict) -> bool:
                base = f"r{state['round_num']:03d}_{tag}_{experiment_id(p)}"
                # Match either the initial run id or a previous retry id for
                # the same semantic config. This avoids retrying every run of
                # the same architecture when only one config failed.
                return any(rid == base or rid.startswith(base + "_retry") for rid in retry_run_ids)
            proposals = [p for p in proposals if _retry_match(p)]
            append_log(f"retry-mode: filtered to {len(proposals)} retry proposals by run_id")
        elif retry_archs:
            # Legacy fallback only; run_id-level retry is preferred.
            proposals = [p for p in proposals if p.get("arch_name") in retry_archs]
            append_log(f"retry-mode: filtered to {len(proposals)} retry proposals by arch")

    fix_mode = state.get("fix_mode", False)

    # If post-codegen review produced a validated subset, submit only that
    # subset. This applies to both normal codegen and fix-mode codegen.
    art_dir = round_artifact_dir(camp, state["round_num"])
    post_review_path = art_dir / "post_codegen_review.json"
    if tag == "smoke" and post_review_path.exists() and not fix_mode and not retry_mode:
        try:
            post_review = json.loads(post_review_path.read_text(encoding="utf-8"))
            # Filter by validated run id / semantic config, not arch_name.  A
            # round can contain multiple configs with the same arch (e.g. two
            # gausres widths), and registered/shared arches must not be allowed
            # back in via manifest_validated_archs unless their exact run id was
            # produced and validated by codegen.
            valid_run_ids: set[str] = set(post_review.get("validated_run_ids", []))
            manifest_path = art_dir / "codegen_manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    valid_run_ids.update(str(r) for r in manifest.get("validated_run_ids", []) if r)
                except json.JSONDecodeError:
                    pass
            if valid_run_ids:
                def _validated_run_match(p: dict) -> bool:
                    rid = f"r{state['round_num']:03d}_{tag}_{experiment_id(p)}"
                    return rid in valid_run_ids
                proposals = [p for p in proposals if _validated_run_match(p)]
                append_log(f"post-codegen review: filtered to {len(proposals)} validated run_ids")
        except json.JSONDecodeError:
            pass

    # For fix-mode submit (smoke repair), filter to exact semantic run ids.
    # Arch-level filtering is unsafe when one round contains multiple configs
    # for the same architecture.
    if fix_mode and tag == "smoke":
        fix_plan_path = round_artifact_dir(camp, state["round_num"]) / "smoke_fix_plan.json"
        manifest_path = round_artifact_dir(camp, state["round_num"]) / "codegen_manifest.json"
        validated_fix_run_ids: set[str] = set()
        validated_fix_base_ids: set[str] = set()
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("mode") == "fix":
                    validated_fix_run_ids = set(manifest.get("validated_run_ids", []))
                    validated_fix_base_ids = {base_run_id(rid) for rid in validated_fix_run_ids}
            except json.JSONDecodeError:
                validated_fix_run_ids = set()
                validated_fix_base_ids = set()
        if fix_plan_path.exists():
            fix_plan = json.loads(fix_plan_path.read_text(encoding="utf-8"))
            fixes_by_base = {}
            for fix in fix_plan.get("fixes", []):
                fid = fix.get("exp_id", "")
                if fix.get("fixable") and fid:
                    fixes_by_base[base_run_id(fid)] = fix
            smoke_cfg_by_run = {
                (r.get("experiment_id") or r.get("exp_id")): (r.get("config") or {})
                for r in state.get("smoke_results", [])
                if r.get("experiment_id") or r.get("exp_id")
            }
            selected: list[dict] = []
            for p in proposals:
                rid = f"r{state['round_num']:03d}_smoke_{experiment_id(p)}"
                bid = base_run_id(rid)
                fix = fixes_by_base.get(bid)
                if not fix:
                    continue
                source_cfg = smoke_cfg_by_run.get(fix.get("exp_id", "")) or p
                fix_type = str(fix.get("fix_type", "code")).lower()
                if "config" in fix_type:
                    patch = fix.get("config_patch") or {}
                    if patch:
                        patched = _apply_safe_config_patch(source_cfg, patch)
                        patched["_repair_source_run_id"] = rid
                        selected.append(patched)
                    continue
                if fix_type == "schema":
                    patch = fix.get("schema_patch") or {}
                    if patch:
                        patched = _apply_schema_config_patch(source_cfg, patch)
                        patched["_repair_base_id"] = base_run_id(fix.get("exp_id", "")) or bid
                        patched["_repair_source_run_id"] = fix.get("exp_id", rid)
                        selected.append(patched)
                    continue
                if fix_type == "code":
                    if rid in validated_fix_run_ids or bid in validated_fix_base_ids:
                        selected.append(dict(source_cfg))

            # A repair may have changed n_c/lr/batch in an earlier attempt,
            # which changes experiment_id and prevents matching any current
            # proposal. For config/schema repairs, fall back to the exact failed
            # run's retained config so the intended repair is not lost.
            seen_sources = {p.get("_repair_source_run_id") for p in selected if p.get("_repair_source_run_id")}
            for fix in fix_plan.get("fixes", []):
                fid = fix.get("exp_id", "")
                if not fix.get("fixable") or not fid or fid in seen_sources:
                    continue
                source_cfg = smoke_cfg_by_run.get(fid)
                if not source_cfg:
                    continue
                fix_type = str(fix.get("fix_type", "code")).lower()
                if "config" in fix_type and fix.get("config_patch"):
                    patched = _apply_safe_config_patch(source_cfg, fix.get("config_patch") or {})
                elif fix_type == "schema" and fix.get("schema_patch"):
                    patched = _apply_schema_config_patch(source_cfg, fix.get("schema_patch") or {})
                else:
                    continue
                patched["_repair_base_id"] = base_run_id(fid)
                patched["_repair_source_run_id"] = fid
                selected.append(patched)
                seen_sources.add(fid)
            proposals = selected
            append_log(f"fix-mode: filtered to {len(proposals)} fixable proposals by run_id/config_patch")

    # For full run, filter to only smoke-passed configs.  When resuming after
    # a repair smoke, do not resubmit semantic configs that already have a
    # completed full result.  Only failed/repaired configs should advance to a
    # new full attempt.
    if tag == "full":
        smoke_results = state.get("smoke_results", [])
        passed_configs: dict[str, dict] = {}
        for r in smoke_results:
            if r.get("status") == "completed":
                cfg = dict(r.get("config") or {})
                if cfg:
                    # Use the smoke-passed config itself as source of truth.
                    # This preserves config repairs such as reduced n_c or
                    # batch_size when advancing to full.  If the collected
                    # result only retained the proposal config, enrich it from
                    # the actual smoke train_config so generated-model
                    # script_path survives smoke -> full promotion.
                    rid = r.get("experiment_id") or r.get("exp_id") or ""
                    smoke_cfg_path = art_dir / "smoke_configs" / f"{rid}.json"
                    if smoke_cfg_path.exists():
                        try:
                            smoke_train_cfg = json.loads(smoke_cfg_path.read_text(encoding="utf-8"))
                            if smoke_train_cfg.get("script_path"):
                                cfg["script_path"] = smoke_train_cfg.get("script_path")
                        except json.JSONDecodeError:
                            pass
                    passed_configs[experiment_id(cfg)] = cfg

        existing_completed: set[str] = set()
        full_results_path = art_dir / "full_results.json"
        if full_results_path.exists():
            try:
                for r in json.loads(full_results_path.read_text(encoding="utf-8")):
                    if r.get("status") == "completed" and r.get("metrics"):
                        cfg = r.get("config") or {}
                        if cfg:
                            existing_completed.add(experiment_id(cfg))
            except json.JSONDecodeError:
                pass
        proposals = [cfg for eid, cfg in passed_configs.items() if eid not in existing_completed]
        if existing_completed:
            append_log(
                f"full dedupe: skipped {len(existing_completed)} already-completed semantic configs"
            )

    if not retry_mode:
        schema_failed: list[dict] = []
        schema_allowed: list[dict] = []
        for cfg in proposals:
            ok_schema, schema_issues = validate_experiment_schema(cfg, stage=f"submit_{tag}")
            if ok_schema:
                schema_allowed.append(cfg)
            else:
                schema_failed.append({
                    "experiment_id": cfg.get("experiment_id") or experiment_id(cfg),
                    "arch_name": cfg.get("arch_name"),
                    "issues": schema_issues,
                    "config": cfg,
                })
        if schema_failed:
            report_path = art_dir / f"{tag}_submit_schema_failures.json"
            report_path.write_text(json.dumps({
                "timestamp": now_iso(),
                "tag": tag,
                "policy": "Hard fail before Condor submit. Use canonical shared.losses LIBRARY names only; aliases such as masked_l1_grad are rejected, not normalized.",
                "failed_count": len(schema_failed),
                "failed": schema_failed,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            append_log(f"submit schema guard: rejected {len(schema_failed)} {tag} candidates before Condor; wrote {report_path.name}")
            proposals = schema_allowed
            if not proposals:
                state["phase"] = "blocked"
                state["status_note"] = f"Submit schema guard blocked all {tag} candidates before Condor; see {report_path.name}"
                return state

    if not retry_mode:
        guarded: list[dict] = []
        allowed: list[dict] = []
        for cfg in proposals:
            guard = resource_feasibility_guard(cfg)
            if guard.get("resource_guard_blocked"):
                guarded.append({
                    "experiment_id": cfg.get("experiment_id") or experiment_id(cfg),
                    "arch_name": cfg.get("arch_name"),
                    "n_c": cfg.get("n_c"),
                    "depth": cfg.get("depth"),
                    "batch_size": cfg.get("batch_size"),
                    **guard,
                })
            else:
                if guard.get("resource_guard_triggered"):
                    cfg.update(guard)
                allowed.append(cfg)
        if guarded:
            report_path = art_dir / f"{tag}_resource_guard_blocked.json"
            report_path.write_text(json.dumps({
                "timestamp": now_iso(),
                "tag": tag,
                "policy": (
                    "Blocked before runner submit; capacity_rationale cannot waive resource feasibility guard. "
                    "Ordinary Hybrid candidates must use batch_size=16. Automatic lower batch sizes are allowed "
                    "only at batch_size=8 for explicit resource_probe/OOM-repair/resource_guard safe-config paths and are "
                    "feasibility evidence, not ordinary leaderboard candidates unless rerun/normalized; batch_size<8 requires manual_resource_probe_approved=True."
                ),
                "blocked_count": len(guarded),
                "blocked": guarded,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            append_log(f"resource guard: blocked {len(guarded)} {tag} ordinary candidates before submit; wrote {report_path.name}")
            proposals = allowed
            if not proposals:
                state["phase"] = "blocked"
                state["status_note"] = f"Resource guard blocked all {tag} candidates before submit; see {report_path.name}"
                return state

    config_dir = art_dir / f"{tag}_configs"
    config_dir.mkdir(exist_ok=True)

    resuming_submit = str(state.get("status_note", "")).startswith(f"{tag} submitting:")
    handles_data = list(state.get(f"{tag}_handles", [])) if resuming_submit else []
    submitted_exp_ids = {h.get("exp_id") for h in handles_data if h.get("exp_id")}
    repair_manifest = load_manifest(camp) if (fix_mode and tag == "smoke") else {"runs": {}}
    for cfg in proposals:
        base_id = cfg.get("_repair_base_id") or experiment_id(cfg)
        if retry_mode:
            entry = retry_manifest.get("runs", {}).get(base_id, {})
            retry_index = int(entry.get("retry_count", 0)) + 1 if entry else int(state.get("retry_index", 1) or 1)
            retry_index = max(1, retry_index)
            exp_id = f"r{state['round_num']:03d}_{tag}_{base_id}_retry{retry_index}"
        elif fix_mode and tag == "smoke":
            entry = repair_manifest.get("runs", {}).get(base_id, {})
            repair_index = int(entry.get("repair_count", 0)) + 1 if entry else 1
            repair_index = max(1, repair_index)
            exp_id = f"r{state['round_num']:03d}_{tag}_{base_id}_repair{repair_index}"
        else:
            exp_id = f"r{state['round_num']:03d}_{tag}_{base_id}"

        if exp_id in submitted_exp_ids:
            append_log(f"  {exp_id}: already submitted in partial submit state, skipping")
            continue

        arch_name = cfg.get("arch_name", "")
        # Get remote paths from executor's RemoteLayout
        exc_layout = exc.remote
        remote_results = f"{exc_layout.runs_dir(exc.campaign_id)}/{exp_id}"
        train_cfg = {
            "experiment_id": exp_id,
            "strategy": f"hybrid_{tag}",
            "arch_name": arch_name,
            "arch_kwargs": {
                "n_c": cfg.get("n_c", 16),
                "depth": cfg.get("depth", 7),
            },
            "loss_name": cfg.get("loss_name", "masked_l1"),
            "loss_kwargs": {},
            "seed": cfg.get("seed", 1),
            "epochs": epochs,
            "lr": cfg.get("lr", 1e-3),
            "batch_size": cfg.get("batch_size", 16),
            "checkpoint_interval": 50,
            "input_features": cfg.get("input_features", "height"),
            "eval_splits": ["val"],
            # Required by train.py TrainConfig schema
            "data_dir": exc_layout.data_dir,
            "results_dir": remote_results,
            "split_manifest_path": exc_layout.split_manifest,
            "heartbeat_interval_epochs": 10,
        }

        schema_patch = cfg.get("_schema_repair_patch") or {}
        if schema_patch:
            arch_kwargs = train_cfg.setdefault("arch_kwargs", {})
            for key in schema_patch.get("arch_kwargs_remove", []) or []:
                arch_kwargs.pop(str(key), None)
            for key, value in (schema_patch.get("arch_kwargs_set", {}) or {}).items():
                arch_kwargs[str(key)] = value

        # Per-run model isolation hard rule: every attempt must provide a local
        # model source to the executor, which uploads it as runs/<exp_id>/model.py
        # and rewrites TrainConfig.script_path to that run-local copy.  Reference
        # models may be copied from shared/models, but jobs must never runtime
        # import shared/models directly.
        model_source = _resolve_model_source_for_attempt(arch_name, cfg, art_dir, state, base_id)
        if not model_source:
            state["phase"] = "blocked"
            state["status_note"] = f"No model source for {exp_id}; per-run model isolation blocked submit"
            save_state(state, camp)
            return state
        train_cfg["script_path"] = str(model_source)
        train_cfg["model_source_kind"] = _model_source_kind(model_source)

        # Save config locally
        (config_dir / f"{exp_id}.json").write_text(
            json.dumps(train_cfg, indent=2), encoding="utf-8")

        # Submit via Condor
        append_log(f"  {exp_id}: submitting via Condor")
        if retry_gpu_requirements:
            handle = exc.submit_gpu_train(
                exp_id, train_cfg,
                gpu_requirements=retry_gpu_requirements,
                request_memory_gb=retry_request_memory_gb,
            )
        else:
            handle = exc.submit_gpu_train(exp_id, train_cfg)
        handles_data.append({
            "exp_id": exp_id,
            "config": cfg,
            "cluster_id": handle.cluster_id,
            "scheduler": handle.scheduler,
            "results_dir": handle.results_dir,
            "remote_results_dir": handle.remote_results_dir,
            "job_name": exp_id,
            "status": handle.status,
            "submitted_at": handle.submitted_at,
        })
        submitted_exp_ids.add(exp_id)
        append_log(f"  {exp_id}: {handle.status} cluster={handle.cluster_id}")

        # CRC/submit-layer failures are not model evidence.  Do not let them
        # silently enter the retry/controller loop as if the experiment ran.
        # Pause and ask for human intervention so the human researcher can restore CRC auth,
        # inspect the login/Condor/file-system issue, then resume explicitly.
        if handle.status == "failed" and not handle.cluster_id:
            state[f"{tag}_handles"] = handles_data
            state["phase"] = "blocked"
            state["submit_tag"] = tag
            state["status_note"] = (
                f"CRC submit/upload failed for {exp_id}; human intervention required"
            )
            save_state(state, camp)
            return state

        # Persist after every submitted job. If a long Condor/H100 submit or
        # external timeout kills the runner mid-batch, the next tick can resume
        # without duplicating completed submissions.
        state[f"{tag}_handles"] = handles_data
        state["phase"] = "submit"
        state["submit_tag"] = tag
        state["status_note"] = f"{tag} submitting: {len(handles_data)}/{len(proposals)} experiments"
        save_state(state, camp)

    state[f"{tag}_handles"] = handles_data
    state["phase"] = "monitor"
    state["submit_time"] = now_iso()
    state["fix_mode"] = False  # clear after submit consumed it
    if retry_mode:
        state["retry_mode"] = False
        state["retry_archs"] = []
        state["retry_run_ids"] = []
        state["retry_reasons"] = {}
        state["retry_index"] = 1
    state["submit_tag"] = tag
    state["status_note"] = f"{tag} submitted: {len(handles_data)} experiments"
    return state


def _build_train_cfg_for_attempt(
    *,
    exp_id: str,
    cfg: dict,
    arch_name: str,
    epochs: int,
    remote_results: str,
    exc: Executor,
    model_source: Path,
) -> dict:
    train_cfg = {
        "experiment_id": exp_id,
        "strategy": "hybrid_smoke" if epochs == SMOKE_EPOCHS else "hybrid_full",
        "arch_name": arch_name,
        "arch_kwargs": {
            "n_c": cfg.get("n_c", 16),
            "depth": cfg.get("depth", 7),
        },
        "loss_name": cfg.get("loss_name", "masked_l1"),
        "loss_kwargs": {},
        "seed": cfg.get("seed", 1),
        "epochs": epochs,
        "lr": cfg.get("lr", 1e-3),
        "batch_size": cfg.get("batch_size", 16),
        "checkpoint_interval": 50,
        "input_features": cfg.get("input_features", "height"),
        "eval_splits": ["val"],
        "data_dir": exc.remote.data_dir,
        "results_dir": remote_results,
        "split_manifest_path": exc.remote.split_manifest,
        "heartbeat_interval_epochs": 10,
        "script_path": str(model_source),
        "model_source_kind": _model_source_kind(model_source),
    }
    schema_patch = cfg.get("_schema_repair_patch") or {}
    if schema_patch:
        arch_kwargs = train_cfg.setdefault("arch_kwargs", {})
        for key in schema_patch.get("arch_kwargs_remove", []) or []:
            arch_kwargs.pop(str(key), None)
        for key, value in (schema_patch.get("arch_kwargs_set", {}) or {}).items():
            arch_kwargs[str(key)] = value
    return train_cfg


def _retry_suffix(run_id: str) -> int | None:
    m = re.search(r"_retry(\d+)$", run_id or "")
    return int(m.group(1)) if m else None


def _observed_retry_ids(state: dict, tag: str, handles_data: list[dict], base_id: str) -> set[str]:
    """Return retry attempt ids observed anywhere in round state.

    Monitor-triggered retry must share the same budget as controller retry, even
    when old retry attempts were collected before monitor accounting existed.
    The artifact/state results are durable evidence and must count toward the
    original manifest retry cap.
    """
    out: set[str] = set()
    pools = [handles_data, state.get(f"{tag}_results", []) or []]
    if tag == "smoke":
        pools.append(state.get("smoke_results", []) or [])
    elif tag == "full":
        pools.append(state.get("full_results", []) or [])
    for pool in pools:
        for item in pool or []:
            rid = item.get("exp_id") or item.get("experiment_id") or ""
            if base_run_id(rid) == base_id and _retry_suffix(rid) is not None:
                out.add(rid)
    return out


def _next_retry_index(base_id: str, handles_data: list[dict], manifest: dict, state: dict | None = None, tag: str = "") -> int:
    max_seen = 0
    for h in handles_data:
        rid = h.get("exp_id", "")
        if base_run_id(rid) != base_id:
            continue
        n = _retry_suffix(rid)
        if n is not None:
            max_seen = max(max_seen, n)
    if state and tag:
        for rid in _observed_retry_ids(state, tag, handles_data, base_id):
            n = _retry_suffix(rid)
            if n is not None:
                max_seen = max(max_seen, n)
    entry = manifest.get("runs", {}).get(base_id, {})
    max_seen = max(max_seen, int(entry.get("retry_count", 0) or 0))
    for a in entry.get("attempts", []) or []:
        rid = a.get("run_id", "")
        n = _retry_suffix(rid)
        if n is not None:
            max_seen = max(max_seen, n)
    return max_seen + 1


def _result_from_terminal_handle(exc: Executor, hd: dict, h: JobHandle) -> dict:
    log_tail = h.log_tail or hd.get("log_tail", "") or ""
    failed_payload = ""
    if h.remote_results_dir:
        try:
            res = exc.run_remote(
                "cat " + __import__("shlex").quote(h.remote_results_dir + "/FAILED") + " 2>/dev/null || true"
            )
            failed_payload = (res.stdout or "").strip() if res.ok else ""
        except Exception:
            failed_payload = ""
    if not log_tail:
        try:
            log_tail = exc.tail_condor_logs(h, 80)
        except Exception:
            log_tail = ""
    combined = "\n".join(x for x in [failed_payload, log_tail] if x)
    cfg = hd.get("config") or {}
    return {
        "exp_id": h.experiment_id,
        "experiment_id": h.experiment_id,
        "arch_name": cfg.get("arch_name"),
        "status": h.status,
        "config": cfg,
        "cluster_id": h.cluster_id,
        "remote_results_dir": h.remote_results_dir,
        "log_tail": combined,
        "error": failed_payload,
    }


def _maybe_submit_monitor_retries(
    *,
    state: dict,
    camp: Path,
    tag: str,
    handles_data: list[dict],
    handles: list[JobHandle],
    exc: Executor,
) -> int:
    """Submit normal retries from monitor without waiting for batch collect.

    This is the same retry semantic used by the controller.  The only difference
    is timing: monitor can launch the next retry as soon as one attempt is
    terminal and deterministically retryable.  Accounting is durable in the same
    attempt manifest: the terminal source attempt is recorded, budget is checked,
    then the submitted retry attempt is recorded immediately.
    """
    terminal_retryable = {"CUDA_OOM", "HIGH_VRAM", "CONDOR_MEMORY_LIMIT", "CONDOR_EVICTED_GPU_DOWNGRADE", "CONDOR_EVICTED", "CONDOR_INTERRUPTED", "CONDOR_HELD", "TRANSIENT_ENV", "MISSING_EVIDENCE_FAIL", "UNKNOWN_TERMINAL_FAIL"}
    terminal_states = {"failed", "evicted", "held", "submit_failed", "missing_metrics"}
    active_retry_bases = {
        base_run_id(hd.get("exp_id", "")) for hd in handles_data
        if "_retry" in hd.get("exp_id", "") and hd.get("status") not in {"completed", "failed", "evicted", "held", "submit_failed", "missing_metrics"}
    }
    manifest = load_manifest(camp)
    art_dir = round_artifact_dir(camp, state["round_num"])
    submitted = 0
    max_per_tick = int(state.get("monitor_retry_max_per_tick", MONITOR_RETRY_MAX_PER_TICK) or MONITOR_RETRY_MAX_PER_TICK)
    for hd, h in list(zip(handles_data, handles)):
        if submitted >= max_per_tick:
            break
        source_id = h.experiment_id
        if h.status not in terminal_states:
            continue
        # A terminal attempt already recorded in manifest has either been
        # handled by monitor-triggered retry or will be handled by controller.
        base_id = base_run_id(source_id)
        entry = ensure_run(manifest, source_id, tag, hd.get("config") or {})
        observed_retry_ids = _observed_retry_ids(state, tag, handles_data, base_id)
        max_retries = int(entry.get("max_retries", 3) or 3)
        if len(observed_retry_ids) >= max_retries:
            # Backward-compatible guard: older monitor-triggered retries may be
            # present in smoke/full results before they were recorded in the
            # manifest. They still consume the original retry budget.
            record_attempt(
                manifest=manifest,
                run_id=source_id,
                run_type=tag,
                attempt_type="retry" if "_retry" in source_id else "initial",
                status=h.status,
                classification="AUTO_FAIL_MAX_RETRIES",
                action="AUTO_FAIL_MAX_RETRIES",
                config=hd.get("config") or {},
                cluster_id=str(h.cluster_id or "") or None,
                evidence=[f"observed_retry_count={len(observed_retry_ids)}", f"max_retries={max_retries}"],
            )
            append_log(f"monitor retry skipped {source_id}: observed retry budget exhausted ({len(observed_retry_ids)}/{max_retries})")
            continue
        if any(a.get("run_id") == source_id and a.get("status") == h.status for a in entry.get("attempts", []) or []):
            continue
        result = _result_from_terminal_handle(exc, hd, h)
        cls = classify_result(result)
        if cls.get("next_action") != "RETRY" or cls.get("classification") not in terminal_retryable:
            continue
        if base_id in active_retry_bases:
            continue

        source_attempt_type = "retry" if "_retry" in source_id else "initial"
        # Record the just-finished source attempt first.  This is what makes
        # monitor-triggered retry share exactly the same durable budget as the
        # controller path.
        entry = record_attempt(
            manifest=manifest,
            run_id=source_id,
            run_type=tag,
            attempt_type=source_attempt_type,
            status=h.status,
            classification=cls.get("classification", "UNKNOWN"),
            action="RETRY",
            config=hd.get("config") or {},
            cluster_id=str(h.cluster_id or "") or None,
            evidence=cls.get("evidence", []),
        )
        limit_status = check_limit(entry, "RETRY", None)
        if limit_status:
            record_attempt(
                manifest=manifest,
                run_id=source_id,
                run_type=tag,
                attempt_type=source_attempt_type,
                status=h.status,
                classification=limit_status,
                action=limit_status,
                config=hd.get("config") or {},
                cluster_id=str(h.cluster_id or "") or None,
                evidence=cls.get("evidence", []),
            )
            append_log(f"monitor retry skipped {source_id}: manifest limit {limit_status}")
            continue

        retry_index = _next_retry_index(base_id, handles_data, manifest, state, tag)
        retry_exp_id = f"r{state['round_num']:03d}_{tag}_{base_id}_retry{retry_index}"
        cfg = dict(hd.get("config") or {})
        arch_name = cfg.get("arch_name", "")
        model_source = _resolve_model_source_for_attempt(arch_name, cfg, art_dir, state, base_id)
        if not model_source:
            append_log(f"monitor retry skipped {source_id}: no model source")
            continue
        remote_results = f"{exc.remote.runs_dir(exc.campaign_id)}/{retry_exp_id}"
        epochs = SMOKE_EPOCHS if tag == "smoke" else FULL_EPOCHS
        train_cfg = _build_train_cfg_for_attempt(
            exp_id=retry_exp_id,
            cfg=cfg,
            arch_name=arch_name,
            epochs=epochs,
            remote_results=remote_results,
            exc=exc,
            model_source=model_source,
        )
        retry_count_before_submit = int(entry.get("retry_count", 0) or 0)
        high_vram = str(cls.get("classification") or "").upper() in MEMORY_RETRY_CLASSES
        if high_vram:
            # CUDA OOM/HIGH_VRAM retries need larger GPU VRAM, not larger host
            # RAM. Keep the default 16GB host memory unless the classifier saw
            # an actual Condor memory-limit failure.
            cls_name = str(cls.get("classification") or "").upper()
            request_memory_gb = 32 if cls_name == "CONDOR_MEMORY_LIMIT" else 16
            handle = exc.submit_gpu_train(
                retry_exp_id,
                train_cfg,
                gpu_requirements='regexp("qa-h100-", Machine) || regexp("qa-a100-", Machine)',
                request_memory_gb=request_memory_gb,
            )
        else:
            handle = exc.submit_gpu_train(retry_exp_id, train_cfg)

        retry_action = "RETRY_SUBMITTED" if handle.status != "failed" else "RETRY_SUBMIT_FAILED"
        record_attempt(
            manifest=manifest,
            run_id=retry_exp_id,
            run_type=tag,
            attempt_type="retry",
            status=handle.status,
            classification=cls.get("classification", "UNKNOWN"),
            action=retry_action,
            config=cfg,
            cluster_id=str(handle.cluster_id or "") or None,
            evidence=cls.get("evidence", []),
        )
        handles_data.append({
            "exp_id": retry_exp_id,
            "config": cfg,
            "cluster_id": handle.cluster_id,
            "scheduler": handle.scheduler,
            "results_dir": handle.results_dir,
            "remote_results_dir": handle.remote_results_dir,
            "job_name": retry_exp_id,
            "status": handle.status,
            "submitted_at": handle.submitted_at,
            "retry_source": source_id,
            "retry_classification": cls.get("classification"),
        })
        active_retry_bases.add(base_id)
        submitted += 1
        append_log(f"monitor retry submitted {retry_exp_id} for {source_id} ({cls.get('classification')}) cluster={handle.cluster_id}")
    save_manifest(camp, manifest)
    if submitted:
        state[f"{tag}_handles"] = handles_data
        save_state(state, camp)
    return submitted


def handle_monitor(state: dict, camp: Path) -> dict:
    """Wait for all jobs to finish (Condor + sentinel polling)."""
    tag = state.get("submit_tag", "smoke")
    handles_data = state.get(f"{tag}_handles", [])
    timeout_min = 45 if tag == "smoke" else 240  # extra buffer for Condor idle queue

    if not handles_data:
        state["phase"] = "failed"
        state["status_note"] = f"No {tag} handles"
        return state

    exc = get_executor(camp)
    ok, ssh_msg = exc.ssh_ok()
    if not ok:
        # CRC availability is not something the autonomous workflow can fix.
        # Pause instead of repeatedly polling or turning an infra outage into
        # model evidence. the human researcher can restore CRC/ControlMaster and resume.
        state["phase"] = "blocked"
        state["status_note"] = (
            f"CRC SSH/ControlMaster unavailable during {tag} monitor; "
            f"human intervention required: {ssh_msg}"
        )
        return state

    # Convert to JobHandle for polling
    handles = []
    for h in handles_data:
        handles.append(JobHandle(
            experiment_id=h["exp_id"],
            cluster_id=h.get("cluster_id"),
            scheduler=h.get("scheduler", "condor"),
            results_dir=h.get("results_dir", ""),
            remote_results_dir=h.get("remote_results_dir", ""),
            status=h.get("status", "submitted"),
            job_name=h.get("job_name", ""),
            submitted_at=h.get("submitted_at", 0.0),
        ))

    handles = exc.poll_handles(handles)

    # H100/A100-only retry starvation rule. OOM retries are escalated to scarce
    # H100/A100 resources. If such a retry remains queued for more than 3h,
    # treat it as terminal failed evidence instead of waiting indefinitely.
    retry_wait_timed_out = []
    now_ts = time.time()
    for h in handles:
        is_retry = "_retry" in (h.experiment_id or "")
        waited_min = (now_ts - float(h.submitted_at or 0.0)) / 60.0 if h.submitted_at else 0.0
        if is_retry and h.status in {"idle", "submitted"} and waited_min >= H100_RETRY_WAIT_TIMEOUT_MIN:
            h.status = "failed"
            h.log_tail = (
                "AUTO_FAIL_H100_RETRY_WAIT_TIMEOUT: retry requiring H100/A100 "
                f"waited {waited_min:.0f} min without starting; marked failed by policy"
            )
            retry_wait_timed_out.append(h)
    if retry_wait_timed_out:
        try:
            exc.cancel_condor(retry_wait_timed_out)
            append_log(f"H100 retry wait timeout: cancelled {len(retry_wait_timed_out)} queued jobs")
        except Exception as e:
            append_log(f"H100 retry wait timeout cancel failed: {e}")

    # Condor held jobs are terminal infrastructure evidence, not active work.
    # Leave the held status intact so failure_classifier routes it to RETRY,
    # but remove the scheduler job now to avoid waiting for the full timeout
    # or accumulating stale held jobs in the queue.
    held_jobs = [h for h in handles if h.status == "held"]
    if held_jobs:
        try:
            exc.cancel_condor(held_jobs)
            append_log(f"held jobs: cancelled {len(held_jobs)} and marked terminal for controller retry")
        except Exception as e:
            append_log(f"held cancel failed: {e}")

    # Sync back
    for h, hd in zip(handles, handles_data):
        hd["status"] = h.status
        if h.log_tail:
            hd["log_tail"] = h.log_tail

    retry_submitted = _maybe_submit_monitor_retries(
        state=state,
        camp=camp,
        tag=tag,
        handles_data=handles_data,
        handles=handles,
        exc=exc,
    )
    if retry_submitted:
        # Rebuild handles so the newly appended retry attempts participate in
        # the same monitor state machine without waiting for batch collection.
        handles = []
        for h in handles_data:
            handles.append(JobHandle(
                experiment_id=h["exp_id"],
                cluster_id=h.get("cluster_id"),
                scheduler=h.get("scheduler", "condor"),
                results_dir=h.get("results_dir", ""),
                remote_results_dir=h.get("remote_results_dir", ""),
                status=h.get("status", "submitted"),
                job_name=h.get("job_name", ""),
                submitted_at=h.get("submitted_at", 0.0),
            ))

    terminal = {"completed", "failed", "evicted", "held", "submit_failed", "missing_metrics"}
    all_done = all(hd["status"] in terminal for hd in handles_data)
    done_count = sum(1 for hd in handles_data if hd["status"] in terminal)

    append_log(f"monitor({tag}): {done_count}/{len(handles_data)} done")

    if all_done:
        state["phase"] = "collect"
        state["status_note"] = f"{tag} done: {done_count}/{len(handles_data)}"
    else:
        # Timeout check
        submit_time = state.get("submit_time")
        if submit_time:
            try:
                st = datetime.fromisoformat(submit_time)
                if st.tzinfo is None:
                    st = st.replace(tzinfo=timezone.utc)
                elapsed = (datetime.now(timezone.utc) - st).total_seconds() / 60.0
                if elapsed > timeout_min:
                    append_log(f"{tag} timeout after {elapsed:.0f}min")
                    # Cancel still-running jobs
                    still_running = [h for h in handles
                                     if h.status not in ("completed", "failed", "evicted", "submit_failed")]
                    if still_running:
                        try:
                            exc.cancel_condor(still_running)
                            append_log(f"cancelled {len(still_running)} zombie jobs")
                        except Exception as e:
                            append_log(f"cancel failed: {e}")
                    state["phase"] = "collect"
                    return state
            except ValueError:
                pass
        state["status_note"] = f"{tag} running: {done_count}/{len(handles_data)}"
    return state


def handle_collect(state: dict, camp: Path) -> dict:
    """Collect metrics from completed jobs."""
    tag = state.get("submit_tag", "smoke")
    handles_data = state.get(f"{tag}_handles", [])
    art_dir = round_artifact_dir(camp, state["round_num"])
    exc = get_executor(camp)

    results = []
    for h in handles_data:
        if h.get("status") != "completed":
            jh = JobHandle(
                experiment_id=h["exp_id"],
                remote_results_dir=h.get("remote_results_dir", ""),
            )
            log_tail = h.get("log_tail", "")
            if not log_tail:
                try:
                    log_tail = exc.tail_condor_logs(jh, 80)
                except Exception:
                    log_tail = ""
            failed_payload = ""
            if h.get("remote_results_dir"):
                try:
                    res = exc.run_remote(
                        "cat " + __import__("shlex").quote(h["remote_results_dir"] + "/FAILED") + " 2>/dev/null || true"
                    )
                    failed_payload = (res.stdout or "").strip() if res.ok else ""
                except Exception:
                    failed_payload = ""
            combined_log = "\n".join(x for x in [failed_payload, log_tail] if x)
            results.append({
                "exp_id": h["exp_id"],
                "arch_name": h.get("config", {}).get("arch_name"),
                "status": h.get("status"),
                "config": h.get("config"),
                "cluster_id": h.get("cluster_id"),
                "remote_results_dir": h.get("remote_results_dir"),
                "log_tail": combined_log,
                "error": failed_payload,
            })
            continue
        jh = JobHandle(
            experiment_id=h["exp_id"],
            remote_results_dir=h.get("remote_results_dir", ""),
        )
        metrics = exc.fetch_remote_metrics(jh)
        result = {
            "exp_id": h["exp_id"],
            "experiment_id": h["exp_id"],
            "arch_name": h.get("config", {}).get("arch_name"),
            "status": "completed",
            "config": h.get("config"),
        }
        cfg = h.get("config") or {}
        arch_kwargs = cfg.get("arch_kwargs") or {}
        for key in ("loss_name", "lr", "batch_size", "input_features", "seed"):
            if key in cfg:
                result[key] = cfg[key]
        for key in ("n_c", "depth"):
            if key in cfg:
                result[key] = cfg[key]
            elif key in arch_kwargs:
                result[key] = arch_kwargs[key]
        if metrics:
            result["metrics"] = metrics
            val_metrics = metrics.get("val_metrics", {})
            r2 = val_metrics.get("r2_median", metrics.get("r2_median", -1))
            result["val_r2_median"] = r2
            # Check NaN loss from train_losses
            train_losses = metrics.get("train_losses", metrics.get("train_loss", []))
            if train_losses and any(v != v or abs(v) > 1e6 for v in train_losses[-5:]):
                result["status"] = "loss_nan"
        else:
            # A Condor job may disappear/terminate without writing metrics.
            # Preserve logs for the controller so it can classify as RETRY
            # (evicted/condor_rm/OOM) or REPAIR instead of treating a missing
            # metrics file as a vague completed run.
            result["status"] = "missing_metrics"
            try:
                result["log_tail"] = exc.tail_condor_logs(jh, 60)
            except Exception:
                result["log_tail"] = h.get("log_tail", "")
        results.append(result)

    # For full runs: merge with existing full_results, append new completions
    # to history, and update best.  Merging protects resume/repair paths from
    # overwriting prior completed evidence with a cancelled duplicate handle.
    artifact_name = "smoke_results.json" if tag == "smoke" else "full_results.json"
    if tag == "full":
        existing: dict[str, dict] = {}
        full_results_path = art_dir / artifact_name
        if full_results_path.exists():
            try:
                for r in json.loads(full_results_path.read_text(encoding="utf-8")):
                    rid = r.get("exp_id") or r.get("experiment_id") or ""
                    if rid:
                        existing[rid] = r
            except json.JSONDecodeError:
                existing = {}
        for r in results:
            rid = r.get("exp_id") or r.get("experiment_id") or ""
            old = existing.get(rid)
            if old and old.get("status") == "completed" and old.get("metrics") and r.get("status") != "completed":
                continue
            if rid:
                existing[rid] = r
        results = list(existing.values())
        state["full_results"] = results

        hpath = history_path(camp)
        for r in results:
            if r.get("status") == "completed" and "metrics" in r:
                with hpath.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                r2 = r.get("val_r2_median", -1)
                if r2 > state.get("best_r2_median", -1):
                    state["best_r2_median"] = r2
    else:
        # Merge with existing smoke_results (don't overwrite â€” retry/fix rounds add to it).
        # Also write the merged view back to the artifact so artifact-only
        # recovery sees the complete smoke round, not just the latest retry.
        existing = {r.get("exp_id"): r for r in state.get("smoke_results", [])}
        for r in results:
            existing[r.get("exp_id", "")] = r
        state["smoke_results"] = list(existing.values())
        results = state["smoke_results"]

    (art_dir / artifact_name).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Deterministic controller owns PASS/WAIT/RETRY/REPAIR/AUTO_FAIL/REVIEW.
    state["last_collect_tag"] = tag
    state["phase"] = "controller"
    state["status_note"] = f"Collected {len(results)} {tag} results â†’ controller"
    return state


# â”€â”€ AI handlers (worker subprocesses) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def handle_propose(state: dict, camp: Path) -> dict:
    append_log("Phase: PROPOSE")
    art_dir = round_artifact_dir(camp, state["round_num"])
    proposals_path = art_dir / "proposals.json"
    if state.get("autonomy") == "full":
        # Avoid silently reusing stale proposals when the planner crashes or
        # returns no new artifact. A failed planner must block the campaign.
        proposals_path.unlink(missing_ok=True)
        ok, out = run_worker("workflow_planner.py", camp)
        append_log(f"planner: ok={ok} out={out[:300]}")
        if not ok:
            state["phase"] = "blocked"
            state["status_note"] = f"Planner worker failed: {out[:500]}"
            return state
    if proposals_path.is_file():
        proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
        if proposals:
            state["proposals"] = proposals
            state["phase"] = "codegen"
            state["status_note"] = f"Proposed {len(proposals)} experiments"
            return state
    state["phase"] = "done"
    state["status_note"] = "No proposals"
    return state


def _record_codegen_failures(camp: Path, state: dict, manifest_data: dict) -> list[str]:
    """Account codegen failures in the shared attempt manifest.

    Initial codegen validation/generation failures are real experiment attempts:
    they consume total-attempt budget.  Follow-up ai_fix codegen failures also
    consume repair budget.  When a limit is exhausted, mark that semantic run as
    terminal AUTO_FAIL_* so later subset continuation does not silently drop it.
    """
    failed_run_ids = list(manifest_data.get("failed_run_ids") or [])
    details = manifest_data.get("failed_details") or []
    if not failed_run_ids:
        return []

    cfg_by_run = {
        f"r{state['round_num']:03d}_smoke_{experiment_id(cfg)}": cfg
        for cfg in state.get("proposals", [])
    }
    detail_by_run = {d.get("run_id"): d for d in details if d.get("run_id")}
    attempt_type = "repair_rerun" if state.get("fix_mode") or manifest_data.get("mode") == "fix" else "initial"
    mf = load_manifest(camp)
    terminal_actions: list[str] = []
    for rid in failed_run_ids:
        cfg = cfg_by_run.get(rid, {})
        detail = detail_by_run.get(rid, {})
        # Codegen repair attempts do not have Condor repair run ids yet, so
        # synthesize a semantic repair attempt id for accounting. base_run_id()
        # strips the suffix, keeping the budget attached to the original config.
        existing = mf.get("runs", {}).get(base_run_id(rid), {})
        # PASS is terminal for accounting but not a failure exhaustion state.
        # A later full failure after a smoke PASS must still be accounted rather
        # than mislabeled as a terminal codegen failure with action PASS.
        existing_status = existing.get("status")
        if existing_status in TERMINAL_STATUSES and str(existing_status).startswith("AUTO_FAIL"):
            terminal_actions.append(f"{rid}:{existing_status}")
            continue
        counted_rid = rid
        if attempt_type == "repair_rerun":
            repair_index = int(existing.get("repair_count", 0)) + 1 if existing else 1
            counted_rid = f"{rid}_repair{repair_index}"
        entry = record_attempt(
            mf,
            counted_rid,
            "smoke",
            attempt_type,
            "failed",
            "CODEGEN_FAILED",
            "REPAIR",
            config=cfg,
            evidence=[detail.get("message") or "codegen failed before submission"],
        )
        limit_action = check_limit(entry, "REPAIR", None)
        if limit_action:
            record_attempt(
                mf,
                counted_rid,
                "smoke",
                attempt_type,
                "failed",
                limit_action,
                limit_action,
                config=cfg,
                evidence=[detail.get("message") or "codegen failure exhausted attempt budget"],
            )
            terminal_actions.append(f"{rid}:{limit_action}")
    save_manifest(camp, mf)
    if terminal_actions:
        append_log("codegen terminal failures: " + ", ".join(terminal_actions))
    return terminal_actions


def handle_codegen(state: dict, camp: Path) -> dict:
    append_log("Phase: CODEGEN")
    if state.get("autonomy") == "full":
        ok, out = run_worker("workflow_codegen.py", camp)
        append_log(f"codegen: ok={ok} out={out[:300]}")
    art_dir = round_artifact_dir(camp, state["round_num"])
    manifest = art_dir / "codegen_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("validated"):
            if state.get("fix_mode") and data.get("mode") == "fix" and int(data.get("generated_count") or 0) == 0:
                fix_plan_path = art_dir / "smoke_fix_plan.json"
                runnable_non_code_fix = False
                if fix_plan_path.exists():
                    try:
                        fix_plan = json.loads(fix_plan_path.read_text(encoding="utf-8"))
                        for fix in fix_plan.get("fixes", []):
                            fix_type = str(fix.get("fix_type", "code")).lower()
                            if fix.get("fixable") and (
                                ("config" in fix_type and fix.get("config_patch"))
                                or (fix_type == "schema" and fix.get("schema_patch"))
                            ):
                                runnable_non_code_fix = True
                                break
                    except json.JSONDecodeError:
                        runnable_non_code_fix = False
                if runnable_non_code_fix:
                    state["codegen_fix_attempt"] = 0
                    state["phase"] = "post_codegen_review"
                    state["submit_tag"] = "smoke"
                    state["status_note"] = "Codegen skipped: non-code repair patch ready â†’ post review"
                    return state

                failed_n = _mark_fix_plan_unrepairable(
                    camp, state,
                    "fix-mode codegen produced zero runnable proposals; proceeding with passed subset",
                )
                state["codegen_fix_attempt"] = 0
                state["fix_mode"] = False
                source_tag = state.get("last_collect_tag") or state.get("submit_tag") or "smoke"
                if source_tag == "full":
                    state["phase"] = "review"
                    state["status_note"] = (
                        f"Fix codegen produced 0 models; terminal-failed {failed_n} targets, proceeding to review"
                    )
                else:
                    state["submit_tag"] = "full"
                    state["phase"] = "submit"
                    state["status_note"] = (
                        f"Fix codegen produced 0 models; terminal-failed {failed_n} targets, "
                        "proceeding to full with passed subset"
                    )
                return state
            state["codegen_fix_attempt"] = 0
            state["phase"] = "post_codegen_review"
            state["submit_tag"] = "smoke"
            # Don't clear fix_mode here â€” handle_submit needs it to filter proposals
            state["status_note"] = f"Codegen OK: {data.get('generated_count')} models â†’ post review"
            return state

        # Codegen can fail for only a subset of proposed experiments. Count
        # those failures in the per-run attempt manifest, including first-pass
        # codegen failures that happen before any Condor submission. Retry up
        # to SMOKE_MAX_FIX_ROUNDS. After attempts are exhausted, mark failed
        # semantic runs terminal in the manifest and continue with the validated
        # subset instead of silently dropping them or blocking the entire round.
        validated_n = int(data.get("validated_count") or 0)
        terminal_failures = _record_codegen_failures(camp, state, data)
        if terminal_failures and validated_n > 0:
            skipped_failed = list(data.get("failed", []))
            data["generated_archs"] = list(data.get("validated_archs", []))
            data["generated_run_ids"] = list(data.get("validated_run_ids", []))
            data["generated_count"] = validated_n
            data["failed_terminal_after_codegen_limits"] = terminal_failures
            data["failed_skipped_after_codegen_retries"] = skipped_failed
            data["failed"] = []
            data["failed_run_ids"] = []
            data["failed_details"] = []
            data["validated"] = True
            manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            state["phase"] = "post_codegen_review"
            state["submit_tag"] = "smoke"
            state["fix_mode"] = False
            state["codegen_fix_attempt"] = 0
            state["status_note"] = (
                f"Codegen limits exhausted; continuing with {validated_n} "
                f"validated models, terminal-failed {len(terminal_failures)}"
            )
            return state

        attempt = int(state.get("codegen_fix_attempt") or state.get("smoke_fix_round", 0) or 0)
        if attempt < SMOKE_MAX_FIX_ROUNDS:
            state["codegen_fix_attempt"] = attempt + 1
            state["phase"] = "ai_fix"
            mode_label = "Fix" if state.get("fix_mode") or data.get("mode") == "fix" else "Initial"
            state["status_note"] = (
                f"{mode_label} codegen partial/failed ({validated_n} validated, "
                f"{len(data.get('failed', []))} failed), retry "
                f"{state['codegen_fix_attempt']}/{SMOKE_MAX_FIX_ROUNDS}"
            )
            return state
        if validated_n > 0:
            skipped_failed = list(data.get("failed", []))
            data["generated_archs"] = list(data.get("validated_archs", []))
            data["generated_run_ids"] = list(data.get("validated_run_ids", []))
            data["generated_count"] = validated_n
            data["failed_skipped_after_codegen_retries"] = skipped_failed
            data["failed"] = []
            data["failed_run_ids"] = []
            data["failed_details"] = []
            data["validated"] = True
            manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            state["phase"] = "post_codegen_review"
            state["submit_tag"] = "smoke"
            state["fix_mode"] = False
            state["codegen_fix_attempt"] = 0
            state["status_note"] = (
                f"Codegen attempts exhausted; continuing with {validated_n} "
                f"validated models, skipping {len(skipped_failed)} failed"
            )
            return state

        state["phase"] = "blocked"
        state["status_note"] = f"Codegen failed: {data.get('failed')}"
        return state
    # No manifest â€” worker may have crashed
    if state.get("autonomy") == "full":
        state["phase"] = "blocked"
        state["status_note"] = "Codegen worker failed (no manifest produced)"
    return state


def handle_post_codegen_review(state: dict, camp: Path) -> dict:
    append_log("Phase: POST_CODEGEN_REVIEW")
    art_dir = round_artifact_dir(camp, state["round_num"])
    review_path = art_dir / "post_codegen_review.json"
    if state.get("autonomy") == "full":
        review_path.unlink(missing_ok=True)
        ok, out = run_worker("workflow_reviewer.py", camp)
        append_log(f"post-codegen reviewer: ok={ok} out={out[:300]}")
        if not ok:
            state["phase"] = "blocked"
            state["status_note"] = f"Post-codegen review worker failed: {out[:500]}"
            return state
    if not review_path.is_file():
        state["phase"] = "blocked"
        state["status_note"] = "Post-codegen review produced no artifact"
        return state
    review = json.loads(review_path.read_text(encoding="utf-8"))
    validated_n = len(review.get("validated_archs", []))
    failed_n = len(review.get("failed", []))
    if review.get("ok"):
        state["phase"] = "submit"
        state["submit_tag"] = "smoke"
        state["status_note"] = f"Post-codegen review OK: {validated_n} validated"
        return state

    # Post-codegen review is separate from smoke retry count, but if it cannot
    # make a model valid, return to ai_fix until the normal smoke/codegen fix
    # budget is exhausted. Then continue with whatever validated subset exists.
    attempt = int(state.get("codegen_fix_attempt") or state.get("smoke_fix_round", 0) or 0)
    if attempt < SMOKE_MAX_FIX_ROUNDS:
        state["codegen_fix_attempt"] = attempt + 1
        state["phase"] = "ai_fix"
        state["status_note"] = (
            f"Post-codegen review failed ({validated_n} validated, {failed_n} failed), "
            f"retry {state['codegen_fix_attempt']}/{SMOKE_MAX_FIX_ROUNDS}"
        )
        return state
    state["phase"] = "blocked"
    state["status_note"] = (
        f"Post-codegen review failed after repair budget "
        f"({validated_n} validated, {failed_n} failed); no Condor submit until review is clean. "
        f"Failures: {review.get('failed')}"
    )
    return state


def handle_controller(state: dict, camp: Path) -> dict:
    append_log("Phase: CONTROLLER")
    art_dir = round_artifact_dir(camp, state["round_num"])
    decision_path = art_dir / "controller_decision.json"
    if state.get("autonomy") == "full":
        decision_path.unlink(missing_ok=True)
        ok, out = run_worker("workflow_controller.py", camp)
        append_log(f"controller: ok={ok} out={out[:300]}")
        if not ok:
            state["phase"] = "blocked"
            state["status_note"] = f"Controller worker failed: {out[:500]}"
            return state
    if not decision_path.is_file():
        state["phase"] = "blocked"
        state["status_note"] = "Controller produced no decision artifact"
        return state
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    state["controller_decision"] = decision
    action = decision.get("round_action")
    tag = decision.get("tag", state.get("last_collect_tag", "smoke"))

    if action == "WAIT":
        state["phase"] = "monitor"
        state["submit_tag"] = tag
        state["status_note"] = "Controller: waiting for in-progress jobs"
    elif action == "FULL_SUBMIT":
        state["phase"] = "submit"
        state["submit_tag"] = "full"
        state["status_note"] = "Controller: smoke terminal â†’ full submit"
    elif action == "ROUND_REVIEW":
        state["phase"] = "review"
        state["status_note"] = "Controller: full terminal â†’ review"
    elif action == "REPAIR":
        # Full-run failures should not send the round back through the smoke
        # repair loop.  Full is the terminal evaluation stage; repairable full
        # failures are recorded in controller_decision and the round proceeds
        # to review with the completed subset.  Otherwise a late full config
        # issue can create Fix round 4/3 and block review indefinitely.
        if tag == "full":
            pass_count = int((decision.get("counts") or {}).get("PASS", 0) or 0)
            if pass_count > 0:
                state["phase"] = "review"
                state["status_note"] = "Controller: full repairable failures skipped â†’ review"
            else:
                state["phase"] = "blocked"
                state["status_note"] = "Controller: full failed with repairable errors and no PASS"
        else:
            state["phase"] = "smoke_classify"
            state["status_note"] = "Controller: repair required â†’ smoke_classify"
    elif action == "RETRY":
        retry_items = [r for r in decision.get("per_run", []) if r.get("action") == "RETRY"]
        retry_archs = [r.get("arch_name") for r in retry_items if r.get("arch_name")]
        retry_run_ids = [r.get("run_id") for r in retry_items if r.get("run_id")]
        retry_reasons = {r.get("run_id"): r.get("classification") for r in retry_items if r.get("run_id")}
        state["retry_mode"] = True
        state["retry_tag"] = tag
        state["retry_archs"] = retry_archs
        state["retry_run_ids"] = retry_run_ids
        state["retry_reasons"] = retry_reasons
        state["retry_index"] = int(state.get("retry_index", 1) or 1)
        state["phase"] = "submit"
        state["submit_tag"] = tag
        state["status_note"] = f"Controller: retry required for {len(retry_run_ids) or len(retry_archs)} runs"
    elif action == "DIAGNOSE":
        state["phase"] = "blocked"
        state["status_note"] = "Controller: needs more evidence/diagnosis"
    else:
        state["phase"] = "blocked"
        state["status_note"] = f"Controller terminal/blocking action: {action}"
    return state


def handle_smoke_classify(state: dict, camp: Path) -> dict:
    append_log("Phase: SMOKE_CLASSIFY")
    if state.get("autonomy") == "full":
        ok, out = run_worker("workflow_reviewer.py", camp)
        append_log(f"smoke reviewer: ok={ok} out={out[:300]}")
    art_dir = round_artifact_dir(camp, state["round_num"])
    fix_plan_path = art_dir / "smoke_fix_plan.json"
    source_tag = state.get("last_collect_tag") or state.get("submit_tag") or "smoke"
    if fix_plan_path.is_file():
        plan = json.loads(fix_plan_path.read_text(encoding="utf-8"))
        source_tag = plan.get("source_tag") or source_tag
        if plan.get("has_fixable"):
            current_round = int(state.get("smoke_fix_round", 0) or 0)
            if current_round >= SMOKE_MAX_FIX_ROUNDS:
                if source_tag == "full":
                    state["phase"] = "review"
                    state["status_note"] = (
                        f"Fix budget exhausted ({current_round}/{SMOKE_MAX_FIX_ROUNDS}); "
                        "proceeding to review"
                    )
                else:
                    state["submit_tag"] = "full"
                    state["phase"] = "submit"
                    state["status_note"] = (
                        f"Fix budget exhausted ({current_round}/{SMOKE_MAX_FIX_ROUNDS}); "
                        "proceeding to full with passed subset"
                    )
                return state
            state["smoke_fix_round"] = current_round + 1
            state["phase"] = "ai_fix"
            state["status_note"] = f"Fix round {state['smoke_fix_round']}/{SMOKE_MAX_FIX_ROUNDS}"
            return state
    # No fixable â€” proceed with what passed, or review if full is already done.
    if source_tag == "full":
        state["phase"] = "review"
        state["status_note"] = "No fixable full failures, proceeding to review"
    else:
        state["submit_tag"] = "full"
        state["phase"] = "submit"
        state["status_note"] = "No fixable failures, proceeding to full run"
    return state


def handle_ai_fix(state: dict, camp: Path) -> dict:
    append_log("Phase: AI_FIX â†’ codegen (fix mode will run next tick)")
    # Mark fix mode in state so codegen can detect it
    state["fix_mode"] = True
    state["phase"] = "codegen"
    state["status_note"] = f"Fix round {state.get('smoke_fix_round', 0)}, codegen will run next tick"
    return state


def handle_review(state: dict, camp: Path) -> dict:
    append_log("Phase: REVIEW")
    art_dir = round_artifact_dir(camp, state["round_num"])
    review_path = art_dir / "round_review.json"
    if state.get("autonomy") == "full":
        # Avoid advancing to next_round on a stale or missing review artifact.
        review_path.unlink(missing_ok=True)
        ok, out = run_worker("workflow_reviewer.py", camp)
        append_log(f"reviewer: ok={ok} out={out[:300]}")
        if not ok:
            state["phase"] = "blocked"
            state["status_note"] = f"Review worker failed: {out[:500]}"
            return state
    if review_path.is_file():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        state["round_review"] = review
        if review.get("action") == "done":
            state["phase"] = "done"
            state["status_note"] = review.get("summary", "Reviewer says done")
            return state
    elif state.get("autonomy") == "full":
        state["phase"] = "blocked"
        state["status_note"] = "Review worker produced no round_review.json"
        return state
    state["phase"] = "next_round"
    state["status_note"] = "Review complete"
    return state


def handle_next_round(state: dict, camp: Path) -> dict:
    append_log("Phase: NEXT_ROUND")
    state["round_num"] += 1
    state["proposals"] = []
    state["smoke_handles"] = []
    state["full_handles"] = []
    state["smoke_results"] = []
    state["smoke_fix_round"] = 0
    state["round_review"] = None
    state["submit_tag"] = ""
    state["fix_mode"] = False
    state["submit_time"] = None
    state["retry_mode"] = False
    state["retry_archs"] = []
    state["retry_run_ids"] = []
    state["retry_reasons"] = {}
    state["retry_index"] = 1
    state["phase"] = "propose"
    state["status_note"] = f"Starting round {state['round_num']}"
    return state


# â”€â”€ Handler dispatch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

HANDLER_MAP = {
    "propose":          handle_propose,
    "codegen":          handle_codegen,
    "post_codegen_review": handle_post_codegen_review,
    "controller":       handle_controller,
    "submit":           handle_submit,     # generic, uses submit_tag
    "monitor":          handle_monitor,    # generic
    "collect":          handle_collect,    # generic
    "smoke_classify":   handle_smoke_classify,
    "ai_fix":           handle_ai_fix,
    "review":           handle_review,
    "next_round":       handle_next_round,
}


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Sequential Campaign Runner")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    camp = Path(args.campaign_dir)
    camp.mkdir(parents=True, exist_ok=True)

    if not acquire_lock():
        print("IDLE locked")
        return

    try:
        # SSH keepalive
        exc = get_executor(camp)
        try:
            exc.ssh_ok()
        except Exception:
            pass

        state = load_state(camp) if args.resume else fresh_state()
        append_log(f"runner start phase={state.get('phase')} round={state.get('round_num')}")

        max_iterations = 15
        for _ in range(max_iterations):
            phase = state.get("phase", "propose")

            if state.get("status") == "paused":
                save_state(state, camp)
                append_log(f"paused at {phase}")
                print(f"IDLE paused")
                return

            if phase in TERMINAL_PHASES:
                save_state(state, camp)
                append_log(f"terminal: {phase}")
                print(f"IDLE {phase}")
                return

            handler = HANDLER_MAP.get(phase)
            if handler is None:
                state["phase"] = "blocked"
                state["status_note"] = f"Unknown phase: {phase}"
                save_state(state, camp)
                return

            old_phase = phase

            # Generic handlers need camp + extra args
            if phase == "submit":
                tag = state.get("submit_tag", "smoke")
                epochs = SMOKE_EPOCHS if tag == "smoke" else FULL_EPOCHS
                state = handle_submit(state, camp, epochs, tag)
            elif phase in ("monitor", "collect"):
                state = handler(state, camp)
            else:
                state = handler(state, camp)

            # Validate transition
            new_phase = state.get("phase")
            if new_phase != old_phase:
                allowed = VALID_TRANSITIONS.get(old_phase, set())
                if new_phase not in allowed:
                    append_log(f"INVALID transition: {old_phase} -> {new_phase}")
                    state["phase"] = "blocked"
                    state["status_note"] = f"Invalid {old_phase}->{new_phase}"
                    save_state(state, camp)
                    return

            save_state(state, camp)

            current = state.get("phase")
            if current in WAIT_PHASES or current in TERMINAL_PHASES:
                break
            if current in MODEL_PHASES:
                break
            append_log(f"chaining: {old_phase} -> {current}")

        append_log(f"runner end phase={state.get('phase')} note={state.get('status_note')}")
        print(json.dumps({
            "phase": state.get("phase"),
            "round": state.get("round_num"),
            "status_note": state.get("status_note"),
        }, ensure_ascii=False, indent=2))
    finally:
        release_lock()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()



