"""Pydantic v2 data contracts for every baseline_source artifact exchanged between modules.

Every on-disk JSON artifact (metrics.json, analysis.json, manifest.json, state.json,
preflight report, codegen log entry) is a serialization of one of these models.
Keeping them in a single file makes the module boundaries explicit and easy to review.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
class ExperimentConfig(BaseModel):
    """Specifies a single experiment (arch + loss + variant + seed + hyperparams)."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    strategy: Literal[
        "grid", "grid_ai_rescue", "ai_explorer",
        "baseline", "stability", "final",
    ]
    arch_name: str
    loss_name: str
    variant: dict[str, Any] = Field(default_factory=dict)
    seed: int
    epochs: int = 500
    lr: float = 1e-3
    batch_size: int = 16
    n_c: int = 16
    phase: Literal["search", "stability", "final"] = "search"
    script_path: Optional[str] = None
    results_dir: Optional[str] = None
    data_dir: Optional[str] = None
    split_manifest_path: Optional[str] = None


class TrainConfig(BaseModel):
    """Arguments to shared.train.train()."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    strategy: str = "unknown"
    seed: int
    epochs: int = 500
    lr: float = 1e-3
    batch_size: int = 16
    checkpoint_interval: int = 50
    arch_name: str = "unet_v3"
    arch_kwargs: dict[str, Any] = Field(default_factory=dict)
    loss_name: str = "masked_l1"
    loss_kwargs: dict[str, Any] = Field(default_factory=dict)
    data_dir: str
    results_dir: str
    split_manifest_path: str
    heartbeat_interval_epochs: int = 10
    compute_r2: bool = False  # per-case RÂ²/MAE at heartbeat (adds ~10s overhead)
    phase: Literal["search", "stability", "final"] = "search"
    eval_splits: list[Literal["val", "holdout"]] = Field(
        default_factory=lambda: ["val"]
    )
    script_path: Optional[str] = None
    # Wall-time early stop rules
    early_stop_wall_min: int = 100  # stop if < baseline RÂ² after this many minutes
    max_wall_min: int = 200  # absolute max wall time in minutes
    baseline_r2_curve_path: Optional[str] = None  # path to {epoch: r2} JSON for same-epoch comparison
    # Input feature configuration
    input_features: Literal["height", "height_sdf", "height_sdf_normal"] = "height"


class EvalConfig(BaseModel):
    """Arguments to EvalModule.evaluate()."""

    model_config = ConfigDict(extra="forbid")

    model_path: str
    split: Literal["val", "holdout"] = "val"
    data_dir: str
    split_manifest_path: str
    arch_name: str
    arch_kwargs: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = 4


# ---------------------------------------------------------------------------
# Metrics + analysis
# ---------------------------------------------------------------------------
class SplitMetrics(BaseModel):
    """Aggregate metrics on one split (val or holdout)."""

    model_config = ConfigDict(extra="forbid")

    r2_median: float
    r2_mean: float
    r2_global: Optional[float] = None
    mae_median: Optional[float] = None
    mae_mean: Optional[float] = None
    per_case_r2: dict[str, float] = Field(default_factory=dict)


class MetricsResult(BaseModel):
    """Canonical schema of metrics.json â€” the sole trainingâ†’analyzer contract."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    strategy: str
    arch_name: str = "unknown"
    loss_name: str = "unknown"
    arch_kwargs: dict[str, Any] = Field(default_factory=dict)
    loss_kwargs: dict[str, Any] = Field(default_factory=dict)
    seed: int
    epochs_trained: int
    wall_time_sec: float
    peak_vram_gb: float
    gpu: str
    status: Literal["ok", "failed", "evicted", "oom"] = "ok"
    val_metrics: SplitMetrics
    holdout_metrics: Optional[SplitMetrics] = None
    config_hash: str = ""
    eval_hash: str = ""
    split_hash: str = ""
    error_message: Optional[str] = None
    early_stop_info: Optional[str] = None  # wall-time early stop reason
    script_path: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


class ExperimentResult(BaseModel):
    """Single-seed experiment result consumed by Analyzer / planners."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    strategy: str
    arch_name: str
    loss_name: str
    arch_kwargs: dict[str, Any] = Field(default_factory=dict)
    loss_kwargs: dict[str, Any] = Field(default_factory=dict)
    seed: int
    val_r2_median: float
    val_r2_mean: Optional[float] = None
    holdout_r2_median: Optional[float] = None
    wall_time_sec: float = 0.0
    peak_vram_gb: float = 0.0
    status: str = "ok"
    metrics_path: Optional[str] = None
    script_path: Optional[str] = None


class AnalysisReport(BaseModel):
    """Per-experiment (multi-seed) aggregation produced by Analyzer."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    arch_name: str
    loss_name: str
    seeds_completed: int
    val_r2_median_stats: dict[str, float]  # mean / std / min / max
    val_r2_median_values: list[float]
    improvement_pct: float
    rank: Optional[int] = None


# ---------------------------------------------------------------------------
# Codegen + preflight
# ---------------------------------------------------------------------------
class CodegenRequest(BaseModel):
    """Request to CodegenService.run()."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    level: Literal["L1", "L2", "L3"]
    parent_template: Optional[str] = None  # L2 only
    target_name: str                       # arch / loss name
    target_kind: Literal["arch", "loss"] = "arch"
    spec: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class CodegenResult(BaseModel):
    """Output of CodegenService.run()."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    success: bool
    generated_files: list[str] = Field(default_factory=list)
    level: str
    rounds_used: int = 0
    preflight_passed: bool = False
    review_passed: bool = False
    error_message: Optional[str] = None
    log_path: Optional[str] = None
    config: Optional[ExperimentConfig] = None


class PreflightCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str = ""


class PreflightReport(BaseModel):
    """Aggregate output of preflight.check()."""

    model_config = ConfigDict(extra="forbid")

    script_path: str
    passed: bool
    checks: list[PreflightCheck] = Field(default_factory=list)
    vram_estimate_gb: Optional[float] = None
    elapsed_sec: float = 0.0


class ReviewAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str
    dimension: str
    status: Literal["accept", "reject", "warn"]
    message: str = ""


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    annotations: list[ReviewAnnotation] = Field(default_factory=list)
    accepted: bool
    merged_feedback: str = ""


# ---------------------------------------------------------------------------
# Campaign state + baseline
# ---------------------------------------------------------------------------
class BaselineMetrics(BaseModel):
    """Baseline reference â€” auto_v2's 7-level UNet, 20 seeds."""

    model_config = ConfigDict(extra="forbid")

    arch_name: str = "unet_v3"
    loss_name: str = "masked_l1"
    r2_median: float = 0.7017
    r2_std: float = 0.005
    seeds: list[int] = Field(default_factory=lambda: [1, 7, 42])
    source: str = "v2 7-level UNet, 20 seeds"


class ChampionSnapshot(BaseModel):
    """Best-known experiment snapshot embedded in CampaignStatus / CampaignState."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    arch_name: str
    loss_name: str
    arch_kwargs: dict[str, Any] = Field(default_factory=dict)
    loss_kwargs: dict[str, Any] = Field(default_factory=dict)
    seed: int
    val_r2_median: float
    val_r2_mean: Optional[float] = None
    holdout_r2_median: Optional[float] = None
    strategy: str
    wall_time_sec: float = 0.0
    script_path: Optional[str] = None


class CampaignState(BaseModel):
    """Atomic campaign-state record written after every planner transition.

    NOTE: ``history.jsonl`` is the single source of truth for experiment
    history. ``state.json`` only records planner mode/state, the champion id,
    and roll-up counters. See ``StateManager.get_history()``.
    """

    model_config = ConfigDict(extra="forbid")

    campaign_id: str
    strategy: str
    config_hash: str = ""
    eval_hash: str = ""
    split_hash: str = ""
    baseline: BaselineMetrics
    planner_mode: str = "grid"
    planner_state: dict[str, Any] = Field(default_factory=dict)
    champion_id: Optional[str] = None
    champion: Optional[ChampionSnapshot] = None
    failure_count: int = 0
    total_submitted: int = 0
    total_completed: int = 0
    rounds_completed: int = 0
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# ---------------------------------------------------------------------------
# Job handles (Condor)
# ---------------------------------------------------------------------------
class JobHandle(BaseModel):
    """Tracks a submitted Condor job through its lifecycle."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    cluster_id: Optional[str] = None
    proc_id: int = 0
    submit_file: str
    results_dir: str
    remote_results_dir: Optional[str] = None
    submitted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    status: Literal[
        "submitted", "idle", "running", "completed",
        "held", "evicted", "failed",
    ] = "submitted"


# ---------------------------------------------------------------------------
# Intermediate campaign status / round report (human-readable, cat-friendly)
# ---------------------------------------------------------------------------
class LeaderboardEntry(BaseModel):
    """One row in the top-N leaderboard written to campaign_status.json."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    arch_name: str
    loss_name: str
    seed: int
    val_r2_median: float
    strategy: str


class CampaignStatus(BaseModel):
    """Running campaign status, refreshed after every runner round.

    Written to ``{campaign_dir}/campaign_status.json`` so the human researcher can ``cat`` it
    at any time and see where the campaign is.
    """

    model_config = ConfigDict(extra="forbid")

    campaign_id: str
    strategy: str
    planner_mode: str = "unknown"
    planner_state: dict[str, Any] = Field(default_factory=dict)
    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    rounds_completed: int = 0
    gpu_hours_used: float = 0.0
    current_best: Optional[ChampionSnapshot] = None
    top5: list[LeaderboardEntry] = Field(default_factory=list)
    last_update: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


class ProposalSnapshot(BaseModel):
    """Minimal proposal summary captured per round."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    arch_name: str
    loss_name: str
    seed: int
    strategy: str
    arch_kwargs: dict[str, Any] = Field(default_factory=dict)
    loss_kwargs: dict[str, Any] = Field(default_factory=dict)


class RoundResultSnapshot(BaseModel):
    """Minimal result summary captured per round."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    status: str
    val_r2_median: Optional[float] = None
    seed: int


class RoundReport(BaseModel):
    """Per-round report written after each planner round."""

    model_config = ConfigDict(extra="forbid")

    campaign_id: str
    strategy: str
    round_number: int
    planner_mode_before: str
    planner_mode_after: str
    proposals: list[ProposalSnapshot] = Field(default_factory=list)
    results: list[RoundResultSnapshot] = Field(default_factory=list)
    champion_changed: bool = False
    previous_champion_id: Optional[str] = None
    new_champion_id: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def dump_json(model: BaseModel, path: str | Path) -> None:
    """Serialize a Pydantic model to disk (non-atomic; callers who need atomicity
    should use StateManager)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(model.model_dump_json(indent=2))


def load_json(cls: type[BaseModel], path: str | Path) -> BaseModel:
    """Load a Pydantic model from its JSON serialization."""
    return cls.model_validate_json(Path(path).read_text())



