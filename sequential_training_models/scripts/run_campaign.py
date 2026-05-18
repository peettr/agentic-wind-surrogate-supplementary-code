"""V4 Campaign Runner — State machine automation loop.

Adapted from V1's proven workflow_runner.py pattern.
Each phase is a独立函数, transitions are validated, state is persisted atomically.

Phases: propose → codegen → submit → monitor → collect → review → next_round → ...
Terminal: done | failed | blocked

Usage:
    python scripts/run_campaign.py --campaign-dir campaigns/round1
    python scripts/run_campaign.py --campaign-dir campaigns/round1 --resume
    python scripts/run_campaign.py --campaign-dir campaigns/round1 --dry-run
"""
from __future__ import annotations

# fcntl removed - using cross-platform PID file lock
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from explorer.codegen import generate_model
from explorer.explorer import ExperimentConfig, ExperimentResult, summarize_results
from explorer.planner_v4 import V4Planner
from explorer.candidate_library import CandidateLibrary
from explorer.reviewer import review_round, build_review_context

LOGGER = logging.getLogger("auto_v4.runner")

# AI binaries

# ======================================================================
# Phase definitions
# ======================================================================

TERMINAL_PHASES = {"done", "failed", "blocked"}

# Valid transitions — any transition not listed here is rejected.
VALID_TRANSITIONS: dict[str, set[str]] = {
    "propose":    {"codegen", "done", "blocked"},
    "codegen":    {"submit", "blocked", "failed"},
    "submit":     {"monitor", "blocked", "failed"},
    "monitor":    {"collect", "blocked"},
    "collect":    {"review", "failed"},
    "review":     {"next_round", "done", "blocked"},
    "next_round": {"propose"},
    "done":       set(),
    "failed":     set(),
    "blocked":    set(),
}

# Phases that require external waiting (don't chain)
WAIT_PHASES = {"monitor", "submit"}

# Phases that run AI/codegen subprocesses (don't chain, let next tick handle)
MODEL_PHASES = {"review", "codegen"}

# ======================================================================
# Lock file (single-instance enforcement)
# ======================================================================

def _lock_path(campaign_dir: Path) -> Path:
    return campaign_dir / "runner.lock"


def acquire_lock(campaign_dir: Path) -> bool:
    """Acquire an exclusive file lock via PID file. Returns False if another runner is active."""
    lock_file = _lock_path(campaign_dir)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    # Check existing lock
    if lock_file.exists():
        try:
            old_pid = int(lock_file.read_text().strip())
            # Check if process is still alive
            if platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x0400, False, old_pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return False
            else:
                os.kill(old_pid, 0)
        except (ValueError, ProcessLookupError, OSError, AttributeError):
            pass  # stale lock, acquire

    try:
        lock_file.write_text(str(os.getpid()))
        return True
    except OSError:
        return False


def release_lock(campaign_dir: Path):
    """Release the lock file."""
    lock_file = _lock_path(campaign_dir)
    if lock_file.exists():
        lock_file.unlink()

# ======================================================================
# State persistence (atomic)
# ======================================================================

def _state_path(campaign_dir: Path) -> Path:
    return campaign_dir / "campaign_state.json"


def save_state(state: dict, campaign_dir: Path) -> None:
    """Atomically save state (tmp + os.replace)."""
    state["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = _state_path(campaign_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_state(campaign_dir: Path) -> dict:
    """Load state or create fresh."""
    path = _state_path(campaign_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "phase": "propose",
        "round_num": 0,
        "proposals": [],
        "handles": [],
        "planner_state": {},
        "best_r2_so_far": -1.0,
        "round_review": None,
        "runner": {},
    }

# ======================================================================
# History (JSONL)
# ======================================================================

def _history_path(campaign_dir: Path) -> Path:
    return campaign_dir / "history.jsonl"


def save_results(results: list[ExperimentResult], campaign_dir: Path) -> None:
    path = _history_path(campaign_dir)
    with path.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "arch_name": r.arch_name,
                "val_r2_median": r.val_r2_median,
                "loss_name": r.loss_name, "lr": r.lr,
                "input_features": r.input_features, "epochs": r.epochs,
                "n_c": r.n_c, "depth": r.depth, "use_ema": r.use_ema, "seed": r.seed,
                "status": r.status, "wall_time_sec": r.wall_time_sec,
                "peak_vram_gb": r.peak_vram_gb, "gpu": r.gpu,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, ensure_ascii=False) + "\n")


def load_history(campaign_dir: Path) -> list[ExperimentResult]:
    path = _history_path(campaign_dir)
    history = []
    if not path.exists():
        return history
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            cfg = ExperimentConfig(
                arch_name=d.get("arch_name", ""),
                n_c=d.get("n_c", 16), depth=d.get("depth", 7),
                loss_name=d.get("loss_name", "masked_l1"),
                lr=d.get("lr", 1e-3),
                input_features=d.get("input_features", "height"),
                epochs=d.get("epochs", 200), seed=d.get("seed", 1),
                use_ema=d.get("use_ema", False),
            )
            r = ExperimentResult(config=cfg, r2_median=d.get("val_r2_median", float("nan")),
                                 status=d.get("status", "pending"))
            r.arch_name = d.get("arch_name", "")
            r.val_r2_median = d.get("val_r2_median")
            r.loss_name = d.get("loss_name", "")
            r.lr = d.get("lr", 1e-3)
            r.input_features = d.get("input_features", "height")
            r.epochs = d.get("epochs", 200)
            r.n_c = d.get("n_c", 16)
            r.depth = d.get("depth", 7)
            r.use_ema = d.get("use_ema", False)
            r.seed = d.get("seed", 1)
            r.wall_time_sec = d.get("wall_time_sec", 0)
            r.peak_vram_gb = d.get("peak_vram_gb", 0)
            r.gpu = d.get("gpu", "")
            history.append(r)
        except Exception:
            pass
    return history

# ======================================================================
# AI integration
# ======================================================================




# ======================================================================
# Condor integration
# ======================================================================

def submit_jobs(configs: list[ExperimentConfig], campaign_dir: Path,
                remote_root: str = "<PROJECT_HPC_ROOT>") -> list[dict]:
    """Generate submit files and submit to Condor. Returns handles."""
    handles = []
    submit_dir = campaign_dir / "submit"
    submit_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        ts = int(time.time()) % 100000
        exp_id = f"{cfg.arch_name}_ep{cfg.epochs}_lr{cfg.lr}_s{cfg.seed}_{ts}"
        results_dir = campaign_dir / "runs" / exp_id
        results_dir.mkdir(parents=True, exist_ok=True)

        # Write train_config.json
        train_cfg = cfg.to_train_config(results_dir)
        train_cfg["experiment_id"] = exp_id
        (results_dir / "train_config.json").write_text(
            json.dumps(train_cfg, indent=2), encoding="utf-8")

        # Submit file
        sub = submit_dir / f"{exp_id}.submit"
        sub.write_text(
            f"executable = {remote_root}/templates/condor_wrapper.sh\n"
            f'arguments = "{remote_root}/shared/train.py {results_dir}/train_config.json"\n'
            f"log = {submit_dir}/{exp_id}.log\n"
            f"error = {submit_dir}/{exp_id}.err\n"
            f"output = {submit_dir}/{exp_id}.out\n"
            f"should_transfer_files = YES\n"
            f"when_to_transfer_output = ON_EXIT\n"
            f"request_memory = 16 GB\n"
            f'requirements = (regexp("qa-h100-", Machine) || regexp("qa-a100-", Machine) || regexp("qa-l40s-", Machine) || regexp("qa-a40-", Machine) || regexp("ta-a6k-", Machine))\n'
            f"queue 1\n",
            encoding="utf-8")

        handles.append({
            "experiment_id": exp_id,
            "config": _cfg_to_dict(cfg),
            "submit_file": str(sub),
            "results_dir": str(results_dir),
            "status": "pending",
        })

    # Submit all
    for h in handles:
        try:
            r = subprocess.run(
                ["condor_submit", h["submit_file"]],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0 and "cluster" in r.stdout.lower():
                import re
                m = re.search(r"(\d+)\s*cluster", r.stdout, re.I)
                if m:
                    h["cluster_id"] = m.group(1)
                    h["status"] = "submitted"
                    LOGGER.info("Submitted %s -> cluster %s", h["experiment_id"], h["cluster_id"])
                else:
                    h["status"] = "submit_failed"
                    LOGGER.warning("Submit output no cluster: %s", r.stdout[:200])
            else:
                h["status"] = "submit_failed"
                LOGGER.warning("condor_submit failed: %s", r.stderr[:200])
        except Exception as e:
            h["status"] = "submit_failed"
            LOGGER.warning("Submit exception for %s: %s", h["experiment_id"], e)

    return handles


def collect_results(handles: list[dict]) -> list[ExperimentResult]:
    """Collect metrics from completed jobs."""
    results = []
    for h in handles:
        if h["status"] not in ("completed",):
            continue
        metrics_path = Path(h["results_dir"]) / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            met = json.loads(metrics_path.read_text(encoding="utf-8"))
            cfg_dict = h["config"]
            cfg = _dict_to_cfg(cfg_dict)
            r = ExperimentResult(
                config=cfg,
                r2_median=met.get("val_metrics", {}).get("r2_median", float("nan")),
                r2_global=met.get("val_metrics", {}).get("r2_global", float("nan")),
                mae_median=met.get("val_metrics", {}).get("mae_median", float("nan")),
                peak_vram_gb=met.get("peak_vram_gb", 0),
                wall_time_sec=met.get("wall_time_sec", 0),
                gpu=met.get("gpu", ""),
                status="ok" if met.get("status") == "ok" else "failed",
            )
            r.arch_name = cfg.arch_name
            r.val_r2_median = r.r2_median
            r.loss_name = cfg.loss_name
            r.lr = cfg.lr
            r.input_features = cfg.input_features
            r.epochs = cfg.epochs
            r.n_c = cfg.n_c
            r.depth = cfg.depth
            r.use_ema = cfg.use_ema
            r.seed = cfg.seed
            results.append(r)
        except Exception as e:
            LOGGER.warning("Collect failed for %s: %s", h["experiment_id"], e)
    return results


# ======================================================================
# Phase handlers
# ======================================================================

def handle_propose(state: dict, camp: Path, planner: V4Planner,
                   history: list[ExperimentResult]) -> dict:
    """Phase: PROPOSE — generate AI prompt, get suggestions, propose experiments."""
    round_num = state["round_num"] + 1
    state["round_num"] = round_num
    LOGGER.info("=== ROUND %d: PROPOSE ===", round_num)

    # Pass review feedback to planner
    review = state.get("round_review")

    # Propose (planner handles 7 scouts + Codex synthesis internally)
    proposals = planner.propose_experiments(history, review)
    if not proposals:
        LOGGER.info("No proposals — campaign complete")
        state["proposals"] = []
        return state

    state["proposals"] = [_cfg_to_dict(p) for p in proposals]
    LOGGER.info("Proposed %d experiments:", len(proposals))
    for p in proposals:
        LOGGER.info("  %s n_c=%d lr=%s loss=%s input=%s ep=%d",
                     p.arch_name, p.n_c, p.lr, p.loss_name, p.input_features, p.epochs)

    return state


def handle_codegen(state: dict, camp: Path) -> dict:
    """Phase: CODEGEN — generate model code for new architectures."""
    LOGGER.info("=== CODEGEN ===")
    proposals = [_dict_to_cfg(d) for d in state["proposals"] if not d.get("_skip")]
    generated = set()

    for cfg in proposals:
        if cfg.arch_name in generated:
            continue
        model_path = PROJECT_ROOT / "models" / "generated" / f"{cfg.arch_name}.py"
        if model_path.exists():
            LOGGER.info("  %s: exists, skip", cfg.arch_name)
            generated.add(cfg.arch_name)
            continue

        LOGGER.info("  Generating %s...", cfg.arch_name)
        result = generate_model(cfg.arch_name, primary_model="codex")
        if result["success"]:
            LOGGER.info("  %s: OK", cfg.arch_name)
            generated.add(cfg.arch_name)
        else:
            LOGGER.error("  %s: FAILED - %s", cfg.arch_name, result["issues"][:3])
            for d in state["proposals"]:
                if d.get("arch_name") == cfg.arch_name:
                    d["_skip"] = True

    return state


def handle_submit(state: dict, camp: Path) -> dict:
    """Phase: SUBMIT — submit jobs to Condor."""
    LOGGER.info("=== SUBMIT ===")
    configs = [_dict_to_cfg(d) for d in state["proposals"] if not d.get("_skip")]
    if not configs:
        LOGGER.info("No configs to submit")
        return state

    handles = submit_jobs(configs, camp)
    state["handles"] = handles
    ok = sum(1 for h in handles if h["status"] == "submitted")
    LOGGER.info("Submitted %d/%d jobs", ok, len(handles))
    return state


def handle_monitor(state: dict, camp: Path) -> dict:
    """Phase: MONITOR — poll until all jobs done."""
    LOGGER.info("=== MONITOR ===")
    handles = state["handles"]
    all_done = True
    for h in handles:
        if h["status"] in ("completed", "failed", "submit_failed"):
            continue
        metrics = Path(h["results_dir"]) / "metrics.json"
        failed = Path(h["results_dir"]) / "FAILED"
        if metrics.exists():
            h["status"] = "completed"
        elif failed.exists():
            h["status"] = "failed"
        else:
            all_done = False

    done_count = sum(1 for h in handles if h["status"] in ("completed", "failed", "submit_failed"))
    LOGGER.info("  %d/%d done", done_count, len(handles))
    if not all_done:
        LOGGER.info("  Still waiting, will check next tick")
    return state


def handle_collect(state: dict, camp: Path) -> dict:
    """Phase: COLLECT — collect metrics from completed jobs."""
    LOGGER.info("=== COLLECT ===")
    handles = state["handles"]
    # Mark remaining as failed if metrics not found
    for h in handles:
        if h["status"] == "completed":
            metrics = Path(h["results_dir"]) / "metrics.json"
            if not metrics.exists():
                h["status"] = "failed"

    results = collect_results(handles)
    save_results(results, camp)

    for r in results:
        if r.val_r2_median is not None and r.val_r2_median > state.get("best_r2_so_far", -1):
            state["best_r2_so_far"] = r.val_r2_median

    LOGGER.info("Collected %d results, best R2=%.4f", len(results), state.get("best_r2_so_far", -1))
    return state


def handle_review(state: dict, camp: Path, planner: V4Planner,
                  history: list[ExperimentResult]) -> dict:
    """Phase: REVIEW — analyze round results."""
    LOGGER.info("=== REVIEW ===")
    round_num = state["round_num"]
    phase_name = planner.phase
    prev_best = state.get("best_r2_so_far", float("-inf"))

    review = review_round(history, round_num, phase_name, prev_best)
    state["round_review"] = review
    LOGGER.info("Review: %s", review.get("summary", ""))

    # Save review
    review_dir = camp / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / f"round_{round_num:04d}.json").write_text(
        json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")

    # Update planner state
    state["planner_state"] = planner.get_state()
    return state


def handle_next_round(state: dict, camp: Path) -> dict:
    """Phase: NEXT_ROUND — prepare for next round."""
    LOGGER.info("=== NEXT ROUND ===")
    state["proposals"] = []
    state["handles"] = []
    state["round_review"] = None
    return state


# ======================================================================
# Handler dispatch
# ======================================================================

def _get_handler(phase: str) -> Callable | None:
    """Get the handler function for a phase."""
    handlers = {
        "propose":    lambda s, c, p, h, ai: handle_propose(s, c, p, h),
        "codegen":    lambda s, c, p, h, ai: handle_codegen(s, c),
        "submit":     lambda s, c, p, h, ai: handle_submit(s, c),
        "monitor":    lambda s, c, p, h, ai: handle_monitor(s, c),
        "collect":    lambda s, c, p, h, ai: handle_collect(s, c),
        "review":     lambda s, c, p, h, ai: handle_review(s, c, p, h),
        "next_round": lambda s, c, p, h, ai: handle_next_round(s, c),
    }
    return handlers.get(phase)

# ======================================================================
# Helpers
# ======================================================================

def _cfg_to_dict(cfg: ExperimentConfig) -> dict:
    return {
        "arch_name": cfg.arch_name, "n_c": cfg.n_c, "depth": cfg.depth,
        "loss_name": cfg.loss_name, "lr": cfg.lr, "batch_size": cfg.batch_size,
        "scheduler": cfg.scheduler, "weight_decay": cfg.weight_decay,
        "gradient_clip": cfg.gradient_clip, "use_ema": cfg.use_ema,
        "ema_decay": cfg.ema_decay, "augmentation": cfg.augmentation,
        "input_features": cfg.input_features, "epochs": cfg.epochs, "seed": cfg.seed,
    }


def _dict_to_cfg(d: dict) -> ExperimentConfig:
    return ExperimentConfig(
        arch_name=d.get("arch_name", ""), n_c=d.get("n_c", 16),
        depth=d.get("depth", 7), loss_name=d.get("loss_name", "masked_l1"),
        lr=d.get("lr", 1e-3), batch_size=d.get("batch_size", 16),
        scheduler=d.get("scheduler"), weight_decay=d.get("weight_decay", 0),
        gradient_clip=d.get("gradient_clip"), use_ema=d.get("use_ema", False),
        ema_decay=d.get("ema_decay", 0.999), augmentation=d.get("augmentation", False),
        input_features=d.get("input_features", "height"),
        epochs=d.get("epochs", 200), seed=d.get("seed", 1),
    )

# ======================================================================
# Main
# ======================================================================

def run_campaign(campaign_dir: str,
                 max_rounds: int = 12, resume: bool = False, dry_run: bool = False):
    """Main entry point: V4 campaign state machine."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    camp = Path(campaign_dir)
    camp.mkdir(parents=True, exist_ok=True)

    if not acquire_lock(camp):
        LOGGER.error("Another runner is active (lock held). Use --campaign-dir for a different dir.")
        return

    try:
        # Init
        library = CandidateLibrary()
        planner = V4Planner(library, camp)
        planner.max_rounds = max_rounds

        state = load_state(camp)
        history = load_history(camp)

        if resume:
            planner.restore_state(state.get("planner_state", {}))
            LOGGER.info("Resumed: phase=%s round=%d history=%d",
                        state["phase"], state["round_num"], len(history))
        else:
            state = load_state(camp)  # fresh

        # Load V3 baseline
        from explorer.explorer import load_v3_baseline
        v3_bl = load_v3_baseline()
        if v3_bl and not any(getattr(r, "arch_name", "") == "unet_v2_baseline" for r in history):
            history.extend(v3_bl)

        LOGGER.info("=== V4 Campaign: %s ===", camp)
        LOGGER.info("max_rounds: %d", max_rounds)

        max_iter = 50  # safety limit
        for _ in range(max_iter):
            phase = state.get("phase", "propose")

            # Terminal
            if phase in TERMINAL_PHASES:
                state.setdefault("runner", {})["last_runner_phase"] = phase
                save_state(state, camp)
                LOGGER.info("Terminal phase: %s", phase)

                # Final summary
                if phase == "done":
                    summary = summarize_results(history)
                    (camp / "final_summary.json").write_text(
                        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                    LOGGER.info("Total: %d experiments", summary["total_experiments"])
                    if summary["top_10"]:
                        best = summary["top_10"][0]
                        LOGGER.info("Best: %s R2=%.4f", best["arch"], best["r2"])
                return

            # Unknown
            handler = _get_handler(phase)
            if handler is None:
                LOGGER.error("Unknown phase: %s", phase)
                state["phase"] = "blocked"
                save_state(state, camp)
                return

            # Execute phase
            LOGGER.info("--- Phase: %s (round %d) ---", phase, state["round_num"])
            state = handler(state, camp, planner, history, None)

            # Reload history after collect
            if phase in ("collect",):
                history = load_history(camp)

            # Save planner state
            state["planner_state"] = planner.get_state()
            state.setdefault("runner", {})["last_runner_phase"] = phase
            save_state(state, camp)

            # Check if propose produced no proposals → done
            if phase == "propose" and not state.get("proposals"):
                state["phase"] = "done"
                save_state(state, camp)
                return

            # Transition validation
            new_phase = state.get("phase")
            if new_phase != phase:
                allowed = VALID_TRANSITIONS.get(phase, set())
                if new_phase not in allowed:
                    LOGGER.error("Invalid transition: %s -> %s (allowed: %s)", phase, new_phase, allowed)
                    state["phase"] = "blocked"
                    save_state(state, camp)
                    return

            # Chain logic
            current = state.get("phase")
            if current in WAIT_PHASES or current in TERMINAL_PHASES:
                # Wait phases need external events (Condor completion)
                LOGGER.info("Waiting phase (%s), stopping until next tick", current)
                break
            if current in MODEL_PHASES:
                LOGGER.info("Model phase (%s), stopping for fresh trigger", current)
                break
            # Chain fast phases (e.g. collect → review → next_round → propose)
            LOGGER.info("Chaining: %s -> %s", phase, current)

    finally:
        release_lock(camp)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="V4 Campaign Runner (state machine)")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_campaign(args.campaign_dir, args.max_rounds, args.resume)


if __name__ == "__main__":
    main()
