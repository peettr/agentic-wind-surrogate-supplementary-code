#!/usr/bin/env python3
"""Rule-based Grid run outcome classifier.

The classifier is intentionally deterministic. It converts collected evidence
into a classification and next action. Ambiguous evidence returns
NEEDS_DIAGNOSIS instead of guessing retry or repair.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ACTION_RECORD_RESULT = "RECORD_RESULT"
ACTION_WAIT = "WAIT"
ACTION_COLLECT_MORE_EVIDENCE = "COLLECT_MORE_EVIDENCE"
ACTION_RETRY = "RETRY"
ACTION_REPAIR = "REPAIR"
ACTION_AUTO_FAIL = "AUTO_FAIL"

PARAM_LIMIT = 150_000_000
H100_RETRY_PARAM_LIMIT = 250_000_000
BLIND_H100_RETRY_PARAM_LIMIT = 400_000_000
H100_QUEUE_TIMEOUT_SEC = 2 * 60 * 60
HIGHEST_TIER = "h100_only_12gb"
SAFEST_TIER = HIGHEST_TIER
HIGH_ACTIVATION_TOKENS = ("fno", "fourier", "spectral", "afno", "ufno", "ffno", "kan", "multiscale", "naf")


@dataclass
class ClassificationResult:
    classification: str
    next_action: str
    confidence: str
    evidence: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    recommended_tier: str | None = None
    recommended_batch_size: int | None = None
    resume_from_checkpoint: bool | None = None
    checkpoint_epoch: int | None = None
    checkpoint_path: str | None = None
    last_epoch: int | None = None
    heartbeat_epoch: int | None = None
    heartbeat_age_sec: int | float | None = None
    walltime_sec: int | float | None = None
    condor_event: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(evidence: dict[str, Any]) -> str:
    metrics = evidence.get("metrics") or {}
    metric_text = ""
    if isinstance(metrics, dict):
        metric_text = "\n".join(str(metrics.get(k, "")) for k in ("error_message", "traceback", "status"))
    return "\n".join(str(evidence.get(k, "")) for k in ("condor_err", "condor_out", "condor_log", "train_log", "error_message")) + "\n" + metric_text


def _is_oom_text(low_text: str) -> bool:
    return (
        "cuda out of memory" in low_text
        or "oom even with batch=8" in low_text
        or "oom with batch=" in low_text
        or ("runtimeerror: cuda" in low_text and "out of memory" in low_text)
    )


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def _is_high_activation(evidence: dict[str, Any]) -> bool:
    haystack = " ".join(str(evidence.get(k, "")).lower() for k in ("arch_name", "run_id", "model_file"))
    return any(tok in haystack for tok in HIGH_ACTIVATION_TOKENS)


def classify_evidence(evidence: dict[str, Any]) -> ClassificationResult:
    text = _text(evidence)
    low = text.lower()

    metrics = evidence.get("metrics") or {}
    if evidence.get("finished") and isinstance(metrics, dict) and metrics.get("status") == "ok":
        if "max_wall_time:" in low:
            return ClassificationResult(
                "MAX_WALL_FINISHED",
                ACTION_RECORD_RESULT,
                "high",
                ["FINISHED exists", "metrics status ok", "train.log contains max_wall_time"],
            )
        if "early_stop:" in low:
            return ClassificationResult(
                "EARLY_STOPPED_BY_BASELINE",
                ACTION_RECORD_RESULT,
                "high",
                ["FINISHED exists", "metrics status ok", "train.log contains early_stop"],
            )
        return ClassificationResult("PASS", ACTION_RECORD_RESULT, "high", ["FINISHED exists", "metrics status ok"])

    if evidence.get("condor_job_status") in {"I", 1, "1"} and not evidence.get("finished") and not evidence.get("failed"):
        idle_evidence = ["condor job is idle", "no training artifacts yet"]
        tier = str(evidence.get("tier") or "")
        idle_seconds = evidence.get("idle_seconds")
        idle_seconds_i = None
        try:
            if idle_seconds is not None:
                idle_seconds_i = int(float(idle_seconds))
                idle_evidence.append(f"idle_seconds={idle_seconds_i}")
        except (TypeError, ValueError):
            idle_seconds_i = None
        if tier == HIGHEST_TIER:
            idle_evidence.append("tier=h100_only_12gb")
            if idle_seconds_i is not None and idle_seconds_i > H100_QUEUE_TIMEOUT_SEC:
                return ClassificationResult(
                    "H100_QUEUE_TIMEOUT",
                    ACTION_AUTO_FAIL,
                    "high",
                    idle_evidence + ["H100-only queue time exceeded 120 minutes"],
                )
        better = evidence.get("better_analyze") or {}
        if isinstance(better, dict):
            able = better.get("able_machines")
            matched = better.get("matched_slots")
            if able is not None:
                idle_evidence.append(f"able_machines={able}")
            if matched is not None:
                idle_evidence.append(f"matched_slots={matched}")
            try:
                if able is not None and int(able) == 0:
                    return ClassificationResult(
                        "IDLE_NEEDS_DIAGNOSIS",
                        ACTION_COLLECT_MORE_EVIDENCE,
                        "high",
                        ["better-analyze no able machines"],
                        ["requirements_or_resources"],
                    )
            except (TypeError, ValueError):
                pass
        idle_gpu = evidence.get("idle_gpu_summary") or {}
        if isinstance(idle_gpu, dict):
            usable = idle_gpu.get("usable_slots")
            matching = idle_gpu.get("matching_tier_slots")
            fragmented = idle_gpu.get("gpu_with_insufficient_cpu_or_memory")
            if usable is not None:
                idle_evidence.append(f"usable_slots={usable}")
            if matching is not None:
                idle_evidence.append(f"matching_tier_slots={matching}")
            if fragmented is not None:
                idle_evidence.append(f"gpu_with_insufficient_cpu_or_memory={fragmented}")
            try:
                if int(fragmented or 0) > 0 and int(usable or 0) > 0:
                    return ClassificationResult("IDLE_RESOURCE_FRAGMENTATION", ACTION_WAIT, "high", idle_evidence)
            except (TypeError, ValueError):
                pass
        return ClassificationResult("IDLE_IN_QUEUE", ACTION_WAIT, "high", idle_evidence)

    if evidence.get("condor_job_status") in {"R", 2, "2"} and not evidence.get("finished") and not evidence.get("failed"):
        return ClassificationResult("RUNNING", ACTION_WAIT, "high", ["condor job is running"])

    if _has_any(text, ("ValidationError", "extra_forbidden", "extra inputs are not permitted", "extra fields not allowed by TrainConfig")):
        return ClassificationResult("SCHEMA_FAIL", ACTION_REPAIR, "high", ["TrainConfig/schema validation error"])

    if _has_any(text, ("SyntaxError",)):
        return ClassificationResult("SYNTAX_FAIL", ACTION_REPAIR, "high", ["SyntaxError in logs"])

    if _has_any(text, ("ImportError", "ModuleNotFoundError", "No module named")):
        return ClassificationResult("IMPORT_FAIL", ACTION_REPAIR, "high", ["import failure in logs"])

    if _has_any(text, ("NameError", "TypeError", "_write_metrics", "missing Model", "shape mismatch")):
        return ClassificationResult("RUNTIME_CODE_FAIL", ACTION_REPAIR, "high", ["code/runtime contract failure in logs"])

    params = evidence.get("params")
    inferred_shape_ok = evidence.get("shape_ok")
    if inferred_shape_ok is None:
        inferred_shape_ok = _shape_ok_from_text(text)
    if params is not None and not _is_oom_text(low):
        try:
            if int(params) > PARAM_LIMIT:
                return ClassificationResult("PARAM_TOO_LARGE", ACTION_REPAIR, "high", [f"params {params} > {PARAM_LIMIT}"])
        except (TypeError, ValueError):
            pass

    if "condor_rm" in low or "job was aborted" in low and "by user" in low:
        return ClassificationResult("CONDOR_INTERRUPTED", ACTION_RETRY, "high", ["Condor log shows user interruption"])

    if "checkpoint_exists" in evidence:
        checkpoint_exists = bool(evidence.get("checkpoint_exists"))
    else:
        checkpoint_exists = bool(evidence.get("checkpoint_path"))
    checkpoint_epoch = evidence.get("checkpoint_epoch")
    heartbeat = evidence.get("heartbeat") or {}
    heartbeat_epoch = evidence.get("heartbeat_epoch")
    if heartbeat_epoch is None and isinstance(heartbeat, dict):
        heartbeat_epoch = heartbeat.get("epoch")
    latest_epoch = evidence.get("latest_train_epoch") or heartbeat_epoch

    if _is_oom_text(low):
        missing = []
        if params is None:
            missing.append("params")
        if inferred_shape_ok is None:
            missing.append("shape_ok")
        if missing:
            return ClassificationResult("NEEDS_DIAGNOSIS", ACTION_COLLECT_MORE_EVIDENCE, "low", ["OOM present but evidence incomplete"], missing)
        try:
            params_i = int(params)
        except (TypeError, ValueError):
            return ClassificationResult("NEEDS_DIAGNOSIS", ACTION_COLLECT_MORE_EVIDENCE, "low", ["OOM present but params invalid"], ["params"])
        tier = str(evidence.get("tier") or "")
        try:
            batch_size = int(evidence.get("batch_size") or 16)
        except (TypeError, ValueError):
            batch_size = 16
        if params_i > PARAM_LIMIT:
            if tier != SAFEST_TIER and params_i <= H100_RETRY_PARAM_LIMIT:
                return ClassificationResult(
                    "PARAM_TOO_LARGE_RETRY_H100",
                    ACTION_RETRY,
                    "high",
                    [
                        "OOM with params above frontend limit on non-H100 tier",
                        f"params={params_i}",
                        f"h100_queue_timeout_sec={H100_QUEUE_TIMEOUT_SEC}",
                    ],
                    recommended_tier=SAFEST_TIER,
                    recommended_batch_size=min(batch_size, 8),
                )
            if params_i > BLIND_H100_RETRY_PARAM_LIMIT:
                return ClassificationResult(
                    "PARAM_TOO_LARGE",
                    ACTION_REPAIR,
                    "high",
                    ["OOM with very large params; downsize before H100 retry", f"params={params_i}"],
                )
            return ClassificationResult("PARAM_TOO_LARGE", ACTION_REPAIR, "high", ["OOM with params above limit", f"params={params_i}"])
        explicit_min_batch_oom = _has_any(text, ("oom even with batch=8", "model too large for gpu"))
        high_tier = tier in {SAFEST_TIER, "h100_a100_l40s_12gb"}
        if explicit_min_batch_oom and batch_size <= 8 and (high_tier or _is_high_activation(evidence)):
            return ClassificationResult(
                "HIGH_VRAM",
                ACTION_REPAIR,
                "high",
                ["CUDA OOM on high-tier at batch_size=8", "runner reported model too large for GPU", f"params={params_i}"],
            )
        if tier != SAFEST_TIER:
            reason = "high activation OOM" if _is_high_activation(evidence) else "CUDA OOM on non-safest tier"
            return ClassificationResult(
                "HIGH_VRAM",
                ACTION_RETRY,
                "high",
                [reason, f"params={params_i}"],
                recommended_tier=SAFEST_TIER,
                recommended_batch_size=min(batch_size, 8),
            )
        if batch_size > 8:
            return ClassificationResult(
                "CUDA_OOM",
                ACTION_RETRY,
                "high",
                ["CUDA OOM on safest tier but batch can be reduced"],
                recommended_tier=SAFEST_TIER,
                recommended_batch_size=8,
            )
        return ClassificationResult("HIGH_VRAM", ACTION_REPAIR, "medium", ["CUDA OOM on safest tier at batch_size=8"])

    timeout_or_eviction = _has_any(text, ("preempt", "evicted", "wall time", "walltime", "time limit", "job was checkpointed", "held by condor"))
    if timeout_or_eviction:
        progress_evidence = []
        if checkpoint_exists:
            progress_evidence.append("checkpoint exists")
        if checkpoint_epoch is not None:
            progress_evidence.append(f"checkpoint_epoch={checkpoint_epoch}")
        if heartbeat_epoch is not None:
            progress_evidence.append(f"heartbeat_epoch={heartbeat_epoch}")
        if latest_epoch is not None:
            progress_evidence.append(f"latest_epoch={latest_epoch}")
        if checkpoint_exists and _has_any(text, ("preempt", "evicted", "job was checkpointed")):
            return ClassificationResult(
                "EVICTED_WITH_CHECKPOINT",
                ACTION_RETRY,
                "high",
                ["Condor log shows eviction/preemption", *progress_evidence],
                recommended_tier=evidence.get("tier"),
                resume_from_checkpoint=True,
                checkpoint_epoch=checkpoint_epoch,
                checkpoint_path=evidence.get("checkpoint_path"),
                last_epoch=latest_epoch,
                heartbeat_epoch=heartbeat_epoch,
                heartbeat_age_sec=evidence.get("heartbeat_age_sec"),
                walltime_sec=evidence.get("walltime_sec"),
                condor_event="evicted_or_preempted",
            )
        if checkpoint_exists or heartbeat_epoch is not None or latest_epoch is not None:
            return ClassificationResult(
                "TIMEOUT_OR_EVICTION_RESUMABLE",
                ACTION_RETRY,
                "medium",
                ["timeout/eviction evidence with progress", *progress_evidence],
                recommended_tier=evidence.get("tier"),
                resume_from_checkpoint=checkpoint_exists,
                checkpoint_epoch=checkpoint_epoch,
                checkpoint_path=evidence.get("checkpoint_path"),
                last_epoch=latest_epoch,
                heartbeat_epoch=heartbeat_epoch,
                heartbeat_age_sec=evidence.get("heartbeat_age_sec"),
                walltime_sec=evidence.get("walltime_sec"),
                condor_event="timeout_or_eviction",
            )
        stale_age = evidence.get("heartbeat_age_sec")
        stale_evidence = ["timeout/eviction evidence without resumable progress"]
        if stale_age is not None:
            stale_evidence.append(f"heartbeat_age_sec={stale_age}")
        return ClassificationResult(
            "STALE_OR_STALLED",
            ACTION_COLLECT_MORE_EVIDENCE,
            "medium",
            stale_evidence,
            ["fresh_heartbeat_or_checkpoint", "latest_train_epoch"],
        )

    if _has_any(text, ("preempt", "evicted")):
        return ClassificationResult("EVICTED", ACTION_RETRY, "medium", ["Condor log shows eviction/preemption"])

    if not text.strip() and not evidence.get("metrics"):
        return ClassificationResult("NEEDS_DIAGNOSIS", ACTION_COLLECT_MORE_EVIDENCE, "low", [], ["logs", "metrics"])

    return ClassificationResult("UNKNOWN_FAIL", ACTION_COLLECT_MORE_EVIDENCE, "low", ["no rule matched"])


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() and path.is_file() else ""


def _read_json(path: Path) -> Any:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text())


def _latest_epoch_from_log(train_log: str) -> int | None:
    epochs = [int(match.group(1)) for match in re.finditer(r"\bEpoch\s+(\d+)\s*/\s*\d+", train_log)]
    return max(epochs) if epochs else None


def _params_from_text(text: str) -> int | None:
    patterns = (
        r"\bparams=(\d+)\b",
        r"\bparams\s+(\d+)\s+limit\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _shape_ok_from_text(text: str) -> bool | None:
    if "FRONTEND_DYNAMIC_OK" in text:
        return True
    if re.search(r"shape\s+(?:mismatch|fail|error)", text, re.IGNORECASE):
        return False
    if re.search(r"Device=.*GPU=.*params=\d+", text):
        return True
    return None


def collect_local_evidence(run_dir: Path, log_dir: Path, precheck_path: Path | None = None) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "finished": (run_dir / "FINISHED").exists(),
        "failed": (run_dir / "FAILED").exists(),
        "metrics": _read_json(run_dir / "metrics.json"),
        "heartbeat": _read_json(run_dir / "HEARTBEAT.json"),
        "train_config": _read_json(run_dir / "train_config.json"),
        "train_log": _read_text(run_dir / "train.log"),
        "condor_err": _read_text(log_dir / "condor.err"),
        "condor_out": _read_text(log_dir / "condor.out"),
        "condor_log": _read_text(log_dir / "condor.log"),
        "checkpoint_exists": (run_dir / "checkpoint.pt").exists(),
        "checkpoint_path": str(run_dir / "checkpoint.pt"),
    }
    if isinstance(evidence.get("heartbeat"), dict):
        evidence["heartbeat_epoch"] = evidence["heartbeat"].get("epoch")
        evidence["heartbeat_time"] = evidence["heartbeat"].get("time")
    latest_epoch = _latest_epoch_from_log(str(evidence.get("train_log") or ""))
    if latest_epoch is not None:
        evidence["latest_train_epoch"] = latest_epoch
    evidence_text = _text(evidence)
    params = _params_from_text(evidence_text)
    if params is not None:
        evidence["params"] = params
    shape_ok = _shape_ok_from_text(evidence_text)
    if shape_ok is not None:
        evidence["shape_ok"] = shape_ok
    if precheck_path is not None and precheck_path.exists():
        evidence["precheck"] = _read_json(precheck_path)
        if isinstance(evidence["precheck"], dict):
            if evidence.get("params") is None and evidence["precheck"].get("params") is not None:
                evidence["params"] = evidence["precheck"].get("params")
            if evidence.get("shape_ok") is None and evidence["precheck"].get("shape_ok") is not None:
                evidence["shape_ok"] = evidence["precheck"].get("shape_ok")
    return evidence



