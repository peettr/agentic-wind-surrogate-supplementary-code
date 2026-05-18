"""Rule-based failure classifier for Hybrid/V5 controller.

AI may be used later for ambiguous diagnosis, but deterministic rules own the
first-pass action: PASS, WAIT, DIAGNOSE, RETRY, REPAIR, AUTO_FAIL.

Unattended-operation bias:
- resource / scheduler / infrastructure failures -> RETRY, bounded by manifest limits
- Python/model/config tracebacks -> REPAIR, bounded by repair limits
- missing evidence -> RETRY once/few times rather than blocking immediately
- DIAGNOSE should be rare and reserved for states with no useful action
"""
from __future__ import annotations

import re
from typing import Any

from workflow_common import resource_feasibility_guard

# Code/config/model failures. These should go to the repair path so the workflow
# can attempt a generated/shared-model fix or eventually skip after repair budget.
CODE_ERROR_PATTERNS = [
    (r"Unknown loss .*masked_l1_grad|LOSS_REGISTRY_KEYERROR", "LOSS_REGISTRY_KEYERROR"),
    (r"INPUT_CHANNEL_MISMATCH|expected input.*to have 1 channels, but got 3 channels", "INPUT_CHANNEL_MISMATCH"),
    (r"ARCH_CONSTRAINT_FAIL|depth must be in .*got", "ARCH_CONSTRAINT_FAIL"),
    (r"from __future__ imports must occur at the beginning|FUTURE_IMPORT_ORDER_FAIL", "FUTURE_IMPORT_ORDER_FAIL"),
    (r"KeyError: .*Unknown arch_name", "CONFIG_ARCH_FAIL"),
    (r"Unknown architecture", "CONFIG_ARCH_FAIL"),
    (r"TrainConfig|validation error|extra_forbidden|Field required", "SCHEMA_FAIL"),
    (r"ValueError:.*num_channels must be divisible by num_groups", "GROUPNORM_FAIL"),
    (r"num_channels must be divisible by num_groups", "GROUPNORM_FAIL"),
    (r"size mismatch|shape mismatch|output shape", "SHAPE_FAIL"),
    (r"mat1 and mat2 shapes cannot be multiplied", "SHAPE_FAIL"),
    (r"The size of tensor .* must match", "SHAPE_FAIL"),
    (r"Given groups=.*expected input", "SHAPE_FAIL"),
    (r"expected .* channels.*got", "SHAPE_FAIL"),
    (r"invalid shape|shape .* is invalid|view size is not compatible", "SHAPE_FAIL"),
    (r"not enough values to unpack|too many values to unpack", "SHAPE_FAIL"),
    (r"einops.*Error", "SHAPE_FAIL"),
    (r"SyntaxError", "SYNTAX_FAIL"),
    (r"ModuleNotFoundError|ImportError", "IMPORT_FAIL"),
    (r"NameError|UnboundLocalError", "NAME_FAIL"),
    (r"TypeError", "TYPE_FAIL"),
    (r"AttributeError", "ATTRIBUTE_FAIL"),
    (r"IndexError", "INDEX_FAIL"),
    (r"AssertionError", "ASSERT_FAIL"),
    (r"NotImplementedError", "NOT_IMPLEMENTED_FAIL"),
    (r"RecursionError|maximum recursion depth", "RECURSION_FAIL"),
    (r"ZeroDivisionError", "ZERO_DIVISION_FAIL"),
    (r"ValueError", "VALUE_FAIL"),
    (r"RuntimeError", "RUNTIME_FAIL"),
]

# Resource/scheduler/transient failures. These should go to retry, bounded by
# manifest limits. CUDA_OOM also triggers runner-side high-memory GPU escalation.
RETRY_PATTERNS = [
    (r"CUDA out of memory|OutOfMemoryError|\bOOM\b|OOM at batch", "CUDA_OOM"),
    (r"CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED", "CUDA_OOM"),
    (r"DefaultCPUAllocator.*can't allocate memory|Cannot allocate memory", "CUDA_OOM"),
    (r"cgroup memory limit|Peak usage: .*megabytes|higher request_memory|RequestMemory", "CONDOR_MEMORY_LIMIT"),
    (r"\bKilled\b|signal 9|out of memory", "CUDA_OOM"),
    (r"evicted|preempted|Job was evicted", "CONDOR_EVICTED"),
    (r"condor_rm|removed", "CONDOR_INTERRUPTED"),
    (r"held|Job was held|HoldReason", "CONDOR_HELD"),
    (r"condor_submit failed|remote file visibility barrier failed|scp failed", "TRANSIENT_ENV"),
    (r"No such file or directory.*conda|node failure|temporary", "TRANSIENT_ENV"),
    (r"Disk quota exceeded|No space left on device|Stale file handle", "TRANSIENT_ENV"),
    (r"SSH unavailable|SSH down|Permission denied", "TRANSIENT_ENV"),
]

SYSTEM_ERROR_HANDLER_PATTERNS = [
    (r"NameError: name '_write_metrics' is not defined", "TRAIN_ERROR_HANDLER_WRITE_METRICS"),
]

HIGH_ACTIVATION_NAMES = (
    "fno", "fourier", "spectral", "afno", "ufno", "ffno", "kan",
    "multiscale", "naf", "attention", "mamba",
)

GPU_DOWNGRADE_OOM_CLASS = "CONDOR_EVICTED_GPU_DOWNGRADE"


def _is_high_mem_gpu(name: str | None) -> bool:
    """Return True for GPUs that should make OOM evidence decisive.

    Host memory and Condor parent-slot Memory are deliberately ignored here.
    Only the GPU attached to the active/OOMing execution segment should count.
    """
    if not name:
        return False
    return bool(re.search(r"\b(H100|A100)\b|80\s*GB|80GB|81251\s*MiB|81344", name, re.IGNORECASE))


def _gpu_at_oom(text: str) -> str | None:
    """Best-effort active GPU for the execution segment that emitted OOM."""
    oom_matches = list(re.finditer(r"CUDA out of memory|OutOfMemoryError|\bOOM\b|OOM at batch", text, re.IGNORECASE))
    end = oom_matches[-1].start() if oom_matches else len(text)
    prefix = text[:end]

    device_matches = list(re.finditer(r"Device=cuda\s+GPU=([^\n]+?)(?:\s+params=|\s+trainable_params=|$)", prefix, re.IGNORECASE))
    if device_matches:
        return device_matches[-1].group(1).strip()

    condor_gpu_matches = list(re.finditer(r"DeviceName\s*=\s*\"([^\"]+)\"", prefix, re.IGNORECASE))
    if condor_gpu_matches:
        return condor_gpu_matches[-1].group(1).strip()
    return None


def _has_prior_high_mem_gpu(text: str, active_gpu: str | None = None) -> bool:
    if not re.search(r"\b(H100|A100)\b|80\s*GB|80GB|81251\s*MiB|81344", text, re.IGNORECASE):
        return False
    if active_gpu and _is_high_mem_gpu(active_gpu):
        return False
    return True


def _has_condor_resume_or_eviction(text: str) -> bool:
    return bool(re.search(r"Resumed from epoch|Job was evicted|\bevicted\b|preempted", text, re.IGNORECASE))


def _is_probe_or_oom_repair_context(cfg: dict[str, Any], guard: dict[str, Any]) -> bool:
    repair_context = str(cfg.get("repair_context") or cfg.get("_repair_context") or "")
    return bool(
        cfg.get("resource_probe")
        or cfg.get("resource_probe_only")
        or cfg.get("manual_resource_probe_approved")
        or cfg.get("manual_probe_only")
        or cfg.get("oom_repair_context")
        or cfg.get("_config_repair_patch")
        or guard.get("resource_probe_only")
        or guard.get("oom_repair_context")
        or guard.get("manual_probe_only")
        or re.search(r"oom|resource[_ -]?probe|manual probe", repair_context, re.IGNORECASE)
    )


def _detail_text(result: dict[str, Any]) -> str:
    """Return diagnostic text excluding the status token itself."""
    parts = [
        str(result.get("log_tail", "")),
        str(result.get("error", "")),
    ]
    metrics = result.get("metrics") or {}
    if metrics.get("error_message"):
        parts.append(str(metrics.get("error_message")))
    return "\n".join(p for p in parts if p and p != "None")


def _text(result: dict[str, Any]) -> str:
    parts = [str(result.get("status", "")), _detail_text(result)]
    return "\n".join(p for p in parts if p)


def _with_system_evidence(evidence: list[str], system_bug_evidence: list[str]) -> list[str]:
    if system_bug_evidence:
        return evidence + system_bug_evidence
    return evidence


def classify_result(result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("status", "")
    arch = (result.get("arch_name") or (result.get("config") or {}).get("arch_name") or "").lower()
    detail_text = _detail_text(result)
    text = _text(result)
    evidence: list[str] = []

    metrics = result.get("metrics") or {}
    metrics_status = str(metrics.get("status") or "").lower()
    metrics_error = str(metrics.get("error_message") or "")
    if status == "completed" and (result.get("metrics") or result.get("val_r2_median") is not None):
        if metrics_status and metrics_status not in {"ok", "completed", "success", "passed"}:
            # Condor can exit normally after train.py writes a metrics.json
            # failure sentinel.  Do not classify these as PASS; continue into
            # retry/repair rules using the metrics error evidence.
            evidence.append(f"metrics.status={metrics_status}")
            if metrics_error:
                evidence.append(f"metrics.error_message={metrics_error}")
        else:
            return {"classification": "PASS", "next_action": "PASS", "confidence": "high", "evidence": ["completed with metrics"], "missing_evidence": []}

    if "AUTO_FAIL_H100_RETRY_WAIT_TIMEOUT" in text:
        return {
            "classification": "AUTO_FAIL_H100_RETRY_WAIT_TIMEOUT",
            "next_action": "AUTO_FAIL_H100_RETRY_WAIT_TIMEOUT",
            "confidence": "high",
            "evidence": ["retry requiring H100/A100 waited more than 3h without starting"],
            "missing_evidence": [],
        }

    if status in {"running", "idle", "submitted"}:
        return {"classification": "IN_PROGRESS", "next_action": "WAIT", "confidence": "high", "evidence": [f"status={status}"], "missing_evidence": []}

    system_bug_evidence: list[str] = []
    for pattern, label in SYSTEM_ERROR_HANDLER_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            system_bug_evidence.append(label)

    # Resource/scheduler failures own retry semantics even if train.py's
    # failure-reporting path subsequently throws a secondary code error.
    for pattern, label in RETRY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            evidence.append(pattern)
            if label == "CUDA_OOM":
                cfg = result.get("config") or {}
                guard = resource_feasibility_guard({**cfg, "arch_name": arch or cfg.get("arch_name")})
                active_gpu = _gpu_at_oom(text)
                if active_gpu:
                    evidence.append(f"active_gpu_at_oom={active_gpu}")
                prior_high_mem_gpu = _has_prior_high_mem_gpu(text, active_gpu)
                if (
                    active_gpu
                    and not _is_high_mem_gpu(active_gpu)
                    and prior_high_mem_gpu
                    and _has_condor_resume_or_eviction(text)
                ):
                    evidence.append("prior H100/A100 training resumed after Condor eviction onto lower-VRAM GPU")
                    return {
                        "classification": GPU_DOWNGRADE_OOM_CLASS,
                        "next_action": "RETRY",
                        "confidence": "high",
                        "evidence": _with_system_evidence(evidence, system_bug_evidence),
                        "missing_evidence": [],
                    }
                batch_size = cfg.get("batch_size")
                try:
                    batch_size = int(batch_size)
                except (TypeError, ValueError):
                    batch_size = None
                at_auto_floor = batch_size is not None and batch_size <= 8
                if (
                    active_gpu
                    and _is_high_mem_gpu(active_gpu)
                    and at_auto_floor
                    and _is_probe_or_oom_repair_context(cfg, guard)
                ):
                    evidence.append("CUDA OOM on active H100/A100 at resource probe/OOM-repair batch_size floor <=8")
                    return {
                        "classification": "AUTO_FAIL_RESOURCE_GUARD",
                        "next_action": "AUTO_FAIL_RESOURCE_GUARD",
                        "confidence": "high",
                        "evidence": _with_system_evidence(evidence, system_bug_evidence),
                        "missing_evidence": [],
                    }
                if any(k in arch for k in HIGH_ACTIVATION_NAMES):
                    label = "HIGH_VRAM"
                    evidence.append(f"high_activation_arch={arch}")
            return {"classification": label, "next_action": "RETRY", "confidence": "high", "evidence": _with_system_evidence(evidence, system_bug_evidence), "missing_evidence": []}

    # Deterministic code/config repair patterns.
    for pattern, label in CODE_ERROR_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            evidence.append(pattern)
            return {"classification": label, "next_action": "REPAIR", "confidence": "high", "evidence": _with_system_evidence(evidence, system_bug_evidence), "missing_evidence": []}

    if status in {"evicted", "held"}:
        label = "CONDOR_MEMORY_LIMIT" if re.search(r"cgroup memory limit|Peak usage: .*megabytes|higher request_memory|RequestMemory", text, re.IGNORECASE) else "CONDOR_HELD"
        return {"classification": label, "next_action": "RETRY", "confidence": "medium", "evidence": [f"status={status}"], "missing_evidence": []}

    if status == "loss_nan":
        return {"classification": "LOSS_NAN", "next_action": "REPAIR", "confidence": "medium", "evidence": ["status=loss_nan"], "missing_evidence": []}

    if status in {"failed", "submit_failed", "missing_metrics"}:
        if not detail_text.strip():
            # Missing evidence is usually a collection/scheduler visibility issue.
            # Retry rather than block; manifest limits prevent infinite loops.
            return {"classification": "MISSING_EVIDENCE_FAIL", "next_action": "RETRY", "confidence": "medium", "evidence": [f"status={status}"], "missing_evidence": ["condor.err/train.log/log_tail"]}
        if re.search(r"Traceback \(most recent call last\)|\b\w+Error\b|\b\w+Exception\b", detail_text):
            return {"classification": "UNCLASSIFIED_RUNTIME_FAIL", "next_action": "REPAIR", "confidence": "medium", "evidence": ["unclassified traceback/error log present"], "missing_evidence": ["specific deterministic pattern"]}
        # There is some evidence but no traceback. Treat as retryable transient
        # instead of blocking unattended runs.
        return {"classification": "UNKNOWN_TERMINAL_FAIL", "next_action": "RETRY", "confidence": "low", "evidence": [f"status={status}", "log present but no known pattern"], "missing_evidence": ["specific deterministic pattern"]}

    return {"classification": "UNKNOWN_STATE", "next_action": "DIAGNOSE", "confidence": "low", "evidence": [f"status={status}"], "missing_evidence": ["terminal marker or logs"]}




