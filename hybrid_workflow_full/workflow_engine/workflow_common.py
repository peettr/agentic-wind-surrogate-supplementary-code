"""Shared utilities for V4 workflow modules.

V4 differences from V1:
- Campaign directory (not flat workspace)
- Multi-experiment per round (batch of ~12 configs)
- Smoke test + AI fix loop before full runs
"""

import json
from datetime import datetime, timezone
from pathlib import Path

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
WORKSPACE = Path(__file__).resolve().parent.parent.parent
AUTO_V6_ROOT = WORKSPACE / "auto_v6"

# CRC
SSH_CONTROL = "<SSH_CONTROL_PATH>"
CRC_HOST = "<HPC_USER>@<HPC_LOGIN>"
CRC_REMOTE_ROOT = "<PROJECT_HPC_ROOT>"
LOG_DIR = f"{CRC_REMOTE_ROOT}/logs"

# AI binaries
CLAUDE_BIN = "claude"
CODEX_BIN = "codex"

# Smoke test
SMOKE_EPOCHS = 20
SMOKE_MAX_FIX_ROUNDS = 3
FULL_EPOCHS = 200

# Experiments per round
EXPERIMENTS_PER_ROUND = 12

# Resource feasibility guard.  Capacity rationale explains why an experiment is
# scientifically interesting; it does not exempt configs that are already known
# to be infeasible as ordinary smoke/full candidates.
RESOURCE_PROBE_SAFE_CONFIG = {
    "batch_size_options": [8],
    "batch_size_policy": "automatic_resource_guard_or_oom_repair_floor_batch8_not_leaderboard_candidate",
    "automatic_batch_size_floor": 8,
    "manual_resource_probe_below_floor_requires": "manual_resource_probe_approved=True",
    "n_c_max": 32,
    "depth_max": 5,
    "techniques": ["batch_size=8", "n_c<=32", "depth<=5"],
    "excluded_without_manual_approval": ["batch_size<8", "AMP", "gradient_checkpointing", "manual_resource_probe"],
    "candidate_semantics": (
        "Automatic lower-batch suggestions stop at batch_size=8 for resource-guard/OOM repair. "
        "batch_size<8 is allowed only as an explicitly approved manual_resource_probe and is not "
        "an ordinary score-seeking Auto V6 candidate or leaderboard candidate."
    ),
}


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _estimated_params(cfg: dict) -> int | None:
    """Return estimated parameter count when a proposal/result carries it."""
    for key in (
        "estimated_params", "estimated_param_count", "param_count", "params",
        "num_params", "n_params", "model_params",
    ):
        value = cfg.get(key)
        if value is None:
            continue
        try:
            if isinstance(value, str):
                text = value.strip().lower().replace(",", "")
                mult = 1
                if text.endswith("b"):
                    mult = 1_000_000_000
                    text = text[:-1]
                elif text.endswith("m"):
                    mult = 1_000_000
                    text = text[:-1]
                return int(float(text) * mult)
            return int(float(value))
        except (TypeError, ValueError):
            continue
    for key in ("estimated_params_million", "params_million", "param_count_million"):
        value = cfg.get(key)
        if value is None:
            continue
        try:
            return int(float(value) * 1_000_000)
        except (TypeError, ValueError):
            continue
    for key in ("estimated_params_billion", "params_billion", "param_count_billion"):
        value = cfg.get(key)
        if value is None:
            continue
        try:
            return int(float(value) * 1_000_000_000)
        except (TypeError, ValueError):
            continue
    return None


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "probe", "resource_probe"}
    return bool(value)


def _has_resource_probe_marker(cfg: dict) -> bool:
    if _truthy(cfg.get("resource_probe")) or _truthy(cfg.get("is_resource_probe")):
        return True
    for key in ("tags", "techniques", "purpose", "primary_purpose", "candidate_type", "mode"):
        value = cfg.get(key)
        if isinstance(value, str) and "resource_probe" in value.lower():
            return True
        if isinstance(value, (list, tuple, set)) and any("resource_probe" in str(v).lower() for v in value):
            return True
    return False


def _has_manual_resource_probe_approval(cfg: dict) -> bool:
    """Explicit the human researcher/manual approval for sub-floor (<8) resource probes only."""
    return _truthy(cfg.get("manual_resource_probe_approved"))


def _has_oom_repair_marker(cfg: dict) -> bool:
    if cfg.get("_config_repair_patch") or cfg.get("_repair_base_id"):
        return True
    for key in ("repair_context", "failure_classification", "next_action", "reason", "status_note"):
        value = str(cfg.get(key) or "").lower()
        if "oom" in value or "out of memory" in value or "resource_guard" in value or "repair" in value:
            return True
    return False


def resource_feasibility_guard(cfg: dict) -> dict:
    """Classify config resource feasibility without mutating it.

    Returns a metadata dict with stable keys for planner artifacts, quality gate
    feedback, runner previews, and retry classification.
    """
    arch = str(cfg.get("arch_name") or "").lower()
    n_c = _coerce_int(cfg.get("n_c"), 16)
    depth = _coerce_int(cfg.get("depth"), 0)
    batch_size = _coerce_int(cfg.get("batch_size"), 16)
    params = _estimated_params(cfg)
    is_probe = _has_resource_probe_marker(cfg)
    is_repair = _has_oom_repair_marker(cfg)
    manual_probe_approved = _has_manual_resource_probe_approval(cfg)
    reasons: list[str] = []
    severity = "allow"
    probe_required = False

    # Auto V6 ordinary candidates are locked to batch_size=16. Automatic lower
    # batches are allowed only at batch_size=8 for explicit resource_probe or
    # OOM/resource-guard repair evidence. batch_size<8 is a manual-only
    # resource-probe path and must never be entered by autonomous workflow.
    if batch_size < 8:
        probe_required = True
        if manual_probe_approved:
            severity = "warn"
            is_probe = True
            reasons.append(
                f"batch_size={batch_size}<8 is allowed only by manual_resource_probe_approved=True; "
                "manual probe only, not a leaderboard candidate"
            )
        else:
            severity = "block"
            reasons.append(
                f"batch_size={batch_size}<8 violates Auto V6 automatic batch8 floor; "
                "resource_probe/OOM repair may not auto-reduce below 8 without manual_resource_probe_approved=True"
            )
    elif batch_size != 16:
        if batch_size == 8 and (is_probe or is_repair):
            probe_required = True
            if severity == "allow":
                severity = "warn"
            reasons.append(
                "batch_size=8 accepted only as resource_probe/OOM/resource_guard repair feasibility evidence; "
                "not an ordinary leaderboard candidate"
            )
        else:
            severity = "block"
            probe_required = True
            reasons.append(
                f"batch_size={batch_size} violates ordinary Auto V6 batch_size=16 lock and automatic batch8 repair floor; "
                "ordinary candidates must use 16 and automatic repairs may only use 8"
            )

    if arch == "cno" and n_c >= 40 and depth >= 6 and batch_size >= 16:
        severity = "block"
        probe_required = True
        reasons.append(
            "CNO n_c>=40 depth>=6 batch_size>=16 is known infeasible as an ordinary smoke/full candidate"
        )

    if params is not None and params > 1_500_000_000 and batch_size > 8:
        severity = "block"
        probe_required = True
        reasons.append(
            f"estimated_params={params} >1.5B with batch_size={batch_size}>8; automatic safe floor is batch_size=8"
        )
    elif params is not None and params > 1_000_000_000 and batch_size >= 16:
        severity = "block" if severity == "block" else "warn"
        probe_required = True
        reasons.append(
            f"estimated_params={params} >1B with batch_size={batch_size}>=16"
        )
    elif params is not None and params > 1_500_000_000:
        severity = "block" if severity == "block" else "warn"
        probe_required = True
        reasons.append(
            f"estimated_params={params} >1.5B requires explicit resource_probe before ordinary full use"
        )

    return {
        "resource_guard_triggered": severity != "allow",
        "resource_guard_severity": severity,
        "resource_guard_blocked": severity == "block",
        "resource_guard_reason": "; ".join(reasons),
        "suggested_safe_config": dict(RESOURCE_PROBE_SAFE_CONFIG),
        "resource_probe_required": probe_required,
        "resource_probe_only": is_probe,
        "oom_repair_context": is_repair,
        "manual_resource_probe_approved": manual_probe_approved,
        "leaderboard_eligible": batch_size == 16 and not is_probe and not is_repair and not manual_probe_approved and severity != "block",
        "manual_probe_only": manual_probe_approved and batch_size < 8,
        "ordinary_batch_size_lock": 16,
        "automatic_batch_size_floor": 8,
        "estimated_params": params,
    }


def annotate_resource_guard(cfg: dict) -> dict:
    """Return a shallow copy of cfg with resource guard metadata attached."""
    annotated = dict(cfg)
    annotated.update(resource_feasibility_guard(cfg))
    return annotated


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def campaign_state_path(campaign_dir: Path) -> Path:
    return campaign_dir / "workflow_state.json"


def load_state(campaign_dir: Path) -> dict:
    p = campaign_state_path(campaign_dir)
    if p.is_file():
        # Some Windows tools write UTF-8 with BOM. Accept it rather than
        # crashing the runner at startup.
        return json.loads(p.read_text(encoding="utf-8-sig"))
    return fresh_state()


def fresh_state() -> dict:
    return {
        "phase": "propose",
        "round_num": 0,
        "smoke_fix_round": 0,
        "proposals": [],
        "smoke_handles": [],
        "full_handles": [],
        "smoke_results": [],
        "history": [],
        "best_r2_median": -1.0,
        "round_review": None,
        "submit_tag": "",
        "fix_mode": False,
        "submit_time": None,
        "status_note": "",
        "autonomy": "full",
    }


def save_state(state: dict, campaign_dir: Path) -> None:
    state["last_update"] = now_iso()
    p = campaign_state_path(campaign_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    import os
    os.replace(tmp, p)


def round_artifact_dir(campaign_dir: Path, round_num: int) -> Path:
    d = campaign_dir / "artifacts" / f"r{round_num:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _id_token(value: object) -> str:
    """Compact filesystem-safe token for experiment IDs."""
    import re

    if value is None:
        return "none"
    text = str(value).strip().lower()
    # Keep common decimal learning-rate notation readable but remove chars that
    # can confuse shell paths or Condor batch names.
    text = re.sub(r"[^a-z0-9.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "none"


def experiment_id(cfg: dict) -> str:
    """Stable semantic ID from config dict.

    The ID is used for smoke/full run directories and Condor batch names.
    Include all fields that commonly define ablations, otherwise configs such
    as nc24-d6 and nc24-d7 collapse to the same run_id and the second proposal
    is skipped as an apparent duplicate.
    """
    name = _id_token(cfg.get("arch_name", "unknown"))
    n_c = _id_token(cfg.get("n_c", 16))
    depth = _id_token(cfg.get("depth", cfg.get("arch_kwargs", {}).get("depth", "na")))
    lr = _id_token(cfg.get("lr", 1e-3))
    loss = _id_token(cfg.get("loss_name", "masked_l1"))
    features = _id_token(cfg.get("input_features", "height"))
    aug = _id_token(cfg.get("augmentation", "none"))
    return f"{name}_nc{n_c}_d{depth}_{features}_{aug}_lr{lr}_{loss}"


def history_path(campaign_dir: Path) -> Path:
    return campaign_dir / "history.jsonl"

