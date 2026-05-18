"""Phase-based runner for auto_v3 campaigns.

Each invocation executes **one phase step**, saves state, and exits.
An external watchdog (cron / Scheduled Task / wrapper script) triggers
repeated invocations until the campaign finishes.

Phase machine::

    init → baseline_submit → baseline_collect →
    search_propose → search_submit → search_collect →
    [search_propose → ... (loop)] →
    finalize

Why phase-based instead of a long-running loop?
- LLM calls (Claude, Codex, GPT) can take minutes and may hang.
  A long-running Python process would block entirely.
- Phase-based execution decouples each step: if one hangs or crashes,
  the watchdog simply re-triggers and the runner picks up where it left off.
- This matches v1's proven approach (Scheduled Task → runner → one step → exit).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

from engine.analyzer import Analyzer
from engine.executor import Executor
from engine.state_manager import KillSwitchError, StateManager
from shared.configs.schema import (
    BaselineMetrics,
    CampaignState,
    CampaignStatus,
    ChampionSnapshot,
    ExperimentConfig,
    ExperimentResult,
    LeaderboardEntry,
    ProposalSnapshot,
    RoundReport,
    RoundResultSnapshot,
)
from shared.eval_module import compute_eval_hash, compute_split_hash
from strategies.base_planner import BasePlanner


LOGGER = logging.getLogger("auto_v3.runner")

FAILURE_RATE_ABORT = 0.5

# Phases that involve LLM/CLI calls (potentially long-running).
LLM_PHASES = {"search_propose", "baseline_submit", "search_submit"}

# All valid phases in execution order.
VALID_PHASES = {
    "init",
    "baseline_submit",
    "baseline_collect",
    "search_propose",
    "search_submit",
    "search_collect",
    "finalize",
}


def _sync_planner_state(state: CampaignState, planner: BasePlanner) -> None:
    """Merge planner-internal state into campaign state before each save.

    Uses update() instead of replace to preserve runner-scoped keys
    (baseline_handles, pending_proposals, etc.) that the runner stores
    in planner_state between phases.
    """
    planner_state = planner.get_state()
    state.planner_state.update(planner_state)
    if hasattr(planner, "mode"):
        state.planner_mode = planner.mode  # type: ignore[attr-defined]


def _config_fingerprint(configs: list[ExperimentConfig]) -> str:
    payload = json.dumps(
        [c.model_dump() for c in configs], sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _select_champion(
    history: list[ExperimentResult],
) -> Optional[ExperimentResult]:
    ok = [
        r for r in history
        if r.status == "ok" and r.val_r2_median is not None
    ]
    if not ok:
        return None
    return max(ok, key=lambda r: r.val_r2_median)


def _metrics_to_results(metrics: list[dict]) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for m in metrics:
        val = m.get("val_metrics") or {}
        holdout = m.get("holdout_metrics") or {}
        out.append(ExperimentResult(
            experiment_id=m.get("experiment_id", "unknown"),
            strategy=m.get("strategy", "unknown"),
            arch_name=m.get("arch_name", "unknown"),
            loss_name=m.get("loss_name", "unknown"),
            arch_kwargs=dict(m.get("arch_kwargs") or {}),
            loss_kwargs=dict(m.get("loss_kwargs") or {}),
            seed=int(m.get("seed", 0)),
            val_r2_median=float(val.get("r2_median", float("nan"))),
            val_r2_mean=float(val.get("r2_mean", float("nan"))),
            holdout_r2_median=(
                float(holdout.get("r2_median"))
                if holdout.get("r2_median") is not None else None
            ),
            wall_time_sec=float(m.get("wall_time_sec", 0.0)),
            peak_vram_gb=float(m.get("peak_vram_gb", 0.0)),
            status=m.get("status", "failed"),
            script_path=m.get("script_path"),
        ))
    return out


def _snapshot_from_result(r: ExperimentResult) -> ChampionSnapshot:
    return ChampionSnapshot(
        experiment_id=r.experiment_id,
        arch_name=r.arch_name,
        loss_name=r.loss_name,
        arch_kwargs=dict(r.arch_kwargs or {}),
        loss_kwargs=dict(r.loss_kwargs or {}),
        seed=r.seed,
        val_r2_median=r.val_r2_median,
        val_r2_mean=r.val_r2_mean,
        holdout_r2_median=r.holdout_r2_median,
        strategy=r.strategy,
        wall_time_sec=r.wall_time_sec,
        script_path=r.script_path,
    )


def _write_campaign_status(
    campaign_dir: Path,
    state: CampaignState,
    history: list[ExperimentResult],
    planner: BasePlanner,
    strategy: str,
) -> None:
    """Write campaign_status.json — human-readable intermediate report."""
    ok = [
        r for r in history
        if r.status == "ok" and r.val_r2_median is not None
    ]
    ok_sorted = sorted(ok, key=lambda r: r.val_r2_median, reverse=True)
    top5 = [
        LeaderboardEntry(
            experiment_id=r.experiment_id,
            arch_name=r.arch_name,
            loss_name=r.loss_name,
            seed=r.seed,
            val_r2_median=r.val_r2_median,
            strategy=r.strategy,
        )
        for r in ok_sorted[:5]
    ]
    current_best = _snapshot_from_result(ok_sorted[0]) if ok_sorted else None
    total_failed = sum(1 for r in history if r.status != "ok")
    gpu_hours = sum(r.wall_time_sec for r in history) / 3600.0
    status_obj = CampaignStatus(
        campaign_id=state.campaign_id,
        strategy=strategy,
        planner_mode=getattr(planner, "mode", getattr(planner, "rounds", "unknown")).__str__(),
        planner_state={
            "rounds": getattr(planner, "rounds", None),
            "stagnation_count": getattr(
                planner, "stagnation_count",
                getattr(planner, "_stagnation", None),
            ),
            "best_r2": getattr(
                planner, "best_r2", getattr(planner, "_last_best", None),
            ),
        },
        total_submitted=state.total_submitted,
        total_completed=state.total_completed,
        total_failed=total_failed,
        rounds_completed=state.rounds_completed,
        gpu_hours_used=gpu_hours,
        current_best=current_best,
        top5=top5,
    )
    (campaign_dir / "campaign_status.json").write_text(
        status_obj.model_dump_json(indent=2)
    )


def _write_round_report(
    campaign_dir: Path,
    state: CampaignState,
    proposals: list[ExperimentConfig],
    batch: list[ExperimentResult],
    planner_mode_before: str,
    planner_mode_after: str,
    champion_changed: bool,
    previous_champion_id: Optional[str],
    new_champion_id: Optional[str],
    round_number: int,
    strategy: str,
) -> None:
    """Write round_report.json — per-round intermediate."""
    report = RoundReport(
        campaign_id=state.campaign_id,
        strategy=strategy,
        round_number=round_number,
        planner_mode_before=planner_mode_before,
        planner_mode_after=planner_mode_after,
        proposals=[
            ProposalSnapshot(
                experiment_id=p.experiment_id,
                arch_name=p.arch_name,
                loss_name=p.loss_name,
                seed=p.seed,
                strategy=p.strategy,
                arch_kwargs=dict((p.variant or {}).get("arch_kwargs", {})),
                loss_kwargs=dict((p.variant or {}).get("loss_kwargs", {})),
            )
            for p in proposals
        ],
        results=[
            RoundResultSnapshot(
                experiment_id=r.experiment_id,
                status=r.status,
                val_r2_median=r.val_r2_median,
                seed=r.seed,
            )
            for r in batch
        ],
        champion_changed=champion_changed,
        previous_champion_id=previous_champion_id,
        new_champion_id=new_champion_id,
    )
    rounds_dir = campaign_dir / "round_reports"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    (rounds_dir / f"round_{round_number:04d}.json").write_text(
        report.model_dump_json(indent=2)
    )
    (campaign_dir / "round_report.json").write_text(
        report.model_dump_json(indent=2)
    )


def _planner_mode(planner: BasePlanner) -> str:
    return str(getattr(planner, "mode", None) or type(planner).__name__)


# ======================================================================
# Phase-based runner
# ======================================================================

def run_step(
    planner: BasePlanner,
    baseline: BaselineMetrics,
    baseline_configs: list[ExperimentConfig],
    campaign_dir: str | Path,
    strategy: str,
    executor: Optional[Executor] = None,
    analyzer: Optional[Analyzer] = None,
    split_manifest_path: str | Path = "shared/data/split_manifest.json",
) -> Optional[str]:
    """Execute ONE phase step. Returns the next phase name, or None if done.

    The caller (watchdog / cron / wrapper) should call this repeatedly
    until it returns None or "finalize".
    """
    campaign_dir = Path(campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)

    executor = executor or Executor(
        campaign_dir, split_manifest_path=split_manifest_path,
    )
    analyzer = analyzer or Analyzer(campaign_dir)
    state_mgr = StateManager(
        campaign_dir, baseline=baseline, strategy=strategy,
    )

    # Check kill switch before doing anything.
    try:
        state_mgr.check_kill_switch()
    except KillSwitchError as exc:
        LOGGER.warning("Kill switch triggered: %s", exc)
        return None

    state = state_mgr.load_or_create()
    phase = state.planner_state.get("_runner_phase", "init") if state.planner_state else "init"

    # Helper to always sync planner state before saving.
    def _save(st: CampaignState, next_phase: Optional[str] = None) -> None:
        _sync_planner_state(st, planner)
        # Store runner phase inside planner_state dict.
        if next_phase is not None:
            st.planner_state["_runner_phase"] = next_phase
        state_mgr.save(st)

    # Load history.
    history: list[ExperimentResult] = state_mgr.get_history()
    LOGGER.info("Phase: %s | history: %d entries", phase, len(history))

    # Restore planner state if resuming.
    planner_state_data = {
        k: v for k, v in (state.planner_state or {}).items()
        if k != "_runner_phase"
    }
    if planner_state_data and history:
        planner.restore_state(planner_state_data)

    # --- PHASE: init --------------------------------------------------
    if phase == "init":
        split_hash = (
            compute_split_hash(split_manifest_path)
            if Path(split_manifest_path).exists()
            else ""
        )
        eval_hash = compute_eval_hash()
        cfg_hash = _config_fingerprint(baseline_configs)
        state_mgr.write_manifest(state, cfg_hash, eval_hash, split_hash)

        if history:
            # Already have results from a previous run — skip baseline.
            LOGGER.info("History exists (%d entries), skipping baseline.", len(history))
            _save(state, next_phase="search_propose")
            return "search_propose"

        _save(state, next_phase="baseline_submit")
        return "baseline_submit"

    # --- PHASE: baseline_submit ---------------------------------------
    if phase == "baseline_submit":
        LOGGER.info("Submitting baseline (%d configs)", len(baseline_configs))
        handles = executor.submit_batch(baseline_configs)
        state.total_submitted += len(handles)
        # Store handle info for collect phase.
        handle_ids = [
            {"experiment_id": h.experiment_id, "cluster_id": h.cluster_id,
             "remote_results_dir": h.remote_results_dir}
            for h in handles
        ]
        state.planner_state["baseline_handles"] = handle_ids
        _save(state, next_phase="baseline_collect")
        return "baseline_collect"

    # --- PHASE: baseline_collect --------------------------------------
    if phase == "baseline_collect":
        # Reconstruct handles from stored info.
        handle_info = state.planner_state.get("baseline_handles", [])
        handles = executor.reconstruct_handles(handle_info, campaign_dir)
        if not handles:
            LOGGER.error("No baseline handles found in state — cannot collect.")
            _save(state, next_phase="finalize")
            return None

        raw = executor.collect_results(handles)
        results = _metrics_to_results(raw)

        n_fail = sum(1 for r in results if r.status != "ok")
        if results and n_fail / len(results) > FAILURE_RATE_ABORT:
            LOGGER.error(
                "Baseline failure rate %.0f%% — aborting.",
                100 * n_fail / len(results),
            )
            for r in results:
                if r.status != "ok":
                    state.failure_count += 1
                state_mgr.append_history(r)
            _save(state, next_phase="finalize")
            return None

        for r in results:
            if r.status != "ok":
                state.failure_count += 1
            state_mgr.append_history(r)
        history = state_mgr.get_history()

        # Record round.
        state.rounds_completed = 1
        state.total_completed = sum(1 for r in results if r.status == "ok")
        new_champ = _select_champion(history)
        if new_champ:
            state_mgr.set_champion(
                state, new_champ.experiment_id,
                snapshot=_snapshot_from_result(new_champ),
            )
        _write_round_report(
            campaign_dir, state, baseline_configs, results,
            "baseline", _planner_mode(planner),
            champion_changed=True,
            previous_champion_id=None,
            new_champion_id=state.champion_id,
            round_number=1, strategy=strategy,
        )
        _write_campaign_status(campaign_dir, state, history, planner, strategy)
        _save(state, next_phase="search_propose")
        return "search_propose"

    # --- PHASE: search_propose ----------------------------------------
    if phase == "search_propose":
        if planner.is_done(history, baseline):
            LOGGER.info("Planner signalled done.")
            _save(state, next_phase="finalize")
            return None

        proposals = planner.propose_experiments(history, baseline)
        if not proposals:
            # Planner needs more results or is waiting — skip this cycle.
            LOGGER.info("No proposals this cycle. Will retry on next trigger.")
            # Don't change phase — stay in search_propose for next invocation.
            _save(state, next_phase="search_propose")
            return "search_propose"

        # Store proposals for submit phase.
        proposal_data = [p.model_dump() for p in proposals]
        state.planner_state["pending_proposals"] = proposal_data
        _save(state, next_phase="search_submit")
        return "search_submit"

    # --- PHASE: search_submit -----------------------------------------
    if phase == "search_submit":
        proposal_data = state.planner_state.get("pending_proposals", [])
        if not proposal_data:
            LOGGER.warning("No pending proposals — back to propose.")
            _save(state, next_phase="search_propose")
            return "search_propose"

        proposals = [ExperimentConfig.model_validate(p) for p in proposal_data]
        LOGGER.info("Submitting %d proposals.", len(proposals))
        handles = executor.submit_batch(proposals)
        state.total_submitted += len(handles)

        handle_ids = [
            {"experiment_id": h.experiment_id, "cluster_id": h.cluster_id,
             "remote_results_dir": h.remote_results_dir}
            for h in handles
        ]
        state.planner_state["search_handles"] = handle_ids
        state.planner_state["search_proposals"] = proposal_data
        _save(state, next_phase="search_collect")
        return "search_collect"

    # --- PHASE: search_collect ----------------------------------------
    if phase == "search_collect":
        handle_info = state.planner_state.get("search_handles", [])
        proposal_data = state.planner_state.get("search_proposals", [])
        proposals = [ExperimentConfig.model_validate(p) for p in proposal_data]

        handles = executor.reconstruct_handles(handle_info, campaign_dir)
        if not handles:
            LOGGER.error("No search handles found — back to propose.")
            _save(state, next_phase="search_propose")
            return "search_propose"

        raw = executor.collect_results(handles)
        batch = _metrics_to_results(raw)

        n_fail = sum(1 for r in batch if r.status != "ok")
        if batch and n_fail / len(batch) > FAILURE_RATE_ABORT:
            LOGGER.error(
                "Failure rate %.0f%% in batch — aborting campaign.",
                100 * n_fail / len(batch),
            )
            state.failure_count += n_fail
            for r in batch:
                state_mgr.append_history(r)
            history = state_mgr.get_history()
            _save(state, next_phase="finalize")
            return None

        for r in batch:
            if r.status != "ok":
                state.failure_count += 1
            state_mgr.append_history(r)
        history = state_mgr.get_history()

        # Record round.
        planner_mode_before = _planner_mode(planner)
        round_number = state.rounds_completed + 1
        state.rounds_completed = round_number
        state.total_completed += sum(1 for r in batch if r.status == "ok")

        prev_champ_id = state.champion_id
        new_champ = _select_champion(history)
        champion_changed = False
        if new_champ and new_champ.experiment_id != prev_champ_id:
            state_mgr.set_champion(
                state, new_champ.experiment_id,
                snapshot=_snapshot_from_result(new_champ),
            )
            champion_changed = True
        else:
            _save(state)

        reports = analyzer.analyze(history, baseline)
        analyzer.write_results(batch, reports)

        _write_round_report(
            campaign_dir, state, proposals, batch,
            planner_mode_before, _planner_mode(planner),
            champion_changed, prev_champ_id, state.champion_id,
            round_number, strategy,
        )
        _write_campaign_status(campaign_dir, state, history, planner, strategy)
        _save(state, next_phase="search_propose")
        return "search_propose"

    # --- PHASE: finalize ----------------------------------------------
    if phase == "finalize":
        champion = _select_champion(history)
        if champion:
            state_mgr.set_champion(
                state, champion.experiment_id,
                snapshot=_snapshot_from_result(champion),
            )
            LOGGER.info(
                "Champion: %s (val R² median = %.4f)",
                champion.experiment_id, champion.val_r2_median,
            )
        _write_campaign_status(campaign_dir, state, history, planner, strategy)
        _save(state, next_phase=None)
        return None

    # Unknown phase — reset.
    LOGGER.warning("Unknown phase '%s', resetting to init.", phase)
    _save(state, next_phase="init")
    return "init"


def run_campaign_loop(
    planner: BasePlanner,
    baseline: BaselineMetrics,
    baseline_configs: list[ExperimentConfig],
    campaign_dir: str | Path,
    strategy: str,
    executor: Optional[Executor] = None,
    analyzer: Optional[Analyzer] = None,
    split_manifest_path: str | Path = "shared/data/split_manifest.json",
    trigger_interval: int = 60,
) -> Optional[ExperimentResult]:
    """Convenience wrapper: run steps in a loop with sleep between.

    This is equivalent to a watchdog calling run_step() every trigger_interval
    seconds. Use this for simple local runs; for production, prefer the
    watchdog pattern (Scheduled Task / cron / wrapper script).
    """
    campaign_dir = Path(campaign_dir)
    campaign_dir.mkdir(parents=True, exist_ok=True)

    executor = executor or Executor(
        campaign_dir, split_manifest_path=split_manifest_path,
    )
    analyzer = analyzer or Analyzer(campaign_dir)

    while True:
        next_phase = run_step(
            planner=planner,
            baseline=baseline,
            baseline_configs=baseline_configs,
            campaign_dir=campaign_dir,
            strategy=strategy,
            executor=executor,
            analyzer=analyzer,
            split_manifest_path=split_manifest_path,
        )
        if next_phase is None:
            break
        LOGGER.info("Step done. Next phase: %s. Sleeping %ds.", next_phase, trigger_interval)
        time.sleep(trigger_interval)

    # Load final state to return champion.
    state_mgr = StateManager(
        campaign_dir, baseline=baseline, strategy=strategy,
    )
    history = state_mgr.get_history()
    return _select_champion(history)


# Backward compat: run_campaign = loop version.
run_campaign = run_campaign_loop

__all__ = ["run_step", "run_campaign", "run_campaign_loop"]
