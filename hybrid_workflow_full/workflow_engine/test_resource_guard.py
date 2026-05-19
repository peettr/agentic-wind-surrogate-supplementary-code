"""No-API tests for Hybrid resource feasibility guard."""
from __future__ import annotations

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from workflow_common import resource_feasibility_guard
from failure_classifier import classify_result
from workflow_reviewer import _safe_config_patch


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_cno_nc44_d6_batch16_blocked() -> None:
    guard = resource_feasibility_guard({"arch_name": "cno", "n_c": 44, "depth": 6, "batch_size": 16})
    _assert(guard["resource_guard_blocked"], "CNO nc44 d6 batch16 should be hard blocked")
    _assert(guard["resource_probe_required"], "blocked CNO should require resource_probe")


def test_cno_nc32_d5_batch16_passes() -> None:
    guard = resource_feasibility_guard({"arch_name": "cno", "n_c": 32, "depth": 5, "batch_size": 16})
    _assert(not guard["resource_guard_triggered"], "CNO nc32 d5 batch16 should pass guard")


def test_195b_batch16_blocked() -> None:
    guard = resource_feasibility_guard({"arch_name": "x", "n_c": 16, "depth": 4, "batch_size": 16, "estimated_params": 1_950_000_000})
    _assert(guard["resource_guard_blocked"], "1.95B params batch16 should be blocked")


def test_high_params_batch8_allowed_resource_probe() -> None:
    guard = resource_feasibility_guard({"arch_name": "x", "n_c": 16, "depth": 4, "batch_size": 8, "estimated_params": 1_950_000_000, "resource_probe": True})
    _assert(not guard["resource_guard_blocked"], "1.95B params batch8 should be allowed as explicit feasibility evidence")
    _assert(guard["resource_probe_required"], "1.95B params batch8 should require resource_probe")
    _assert(guard["resource_guard_severity"] == "warn", "1.95B params batch8 should be warning/probe")
    _assert(not guard["leaderboard_eligible"], "batch8 resource probe must not be leaderboard eligible")


def test_ordinary_batch8_without_probe_blocked() -> None:
    guard = resource_feasibility_guard({"arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 8})
    _assert(guard["resource_guard_blocked"], "ordinary batch_size=8 should be blocked by batch-size lock")
    _assert("batch_size=8" in guard["resource_guard_reason"], f"unexpected reason: {guard}")


def test_resource_probe_batch4_blocked_without_manual_approval() -> None:
    guard = resource_feasibility_guard({"arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 4, "resource_probe": True})
    _assert(guard["resource_guard_blocked"], "resource_probe batch_size=4 should be blocked without manual approval")
    _assert("batch_size=4<8" in guard["resource_guard_reason"], f"unexpected reason: {guard}")


def test_manual_resource_probe_batch4_allowed_non_leaderboard() -> None:
    guard = resource_feasibility_guard({
        "arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 4,
        "resource_probe": True, "manual_resource_probe_approved": True,
    })
    _assert(not guard["resource_guard_blocked"], "manual_resource_probe_approved batch_size=4 should be allowed")
    _assert(guard["manual_probe_only"], "manual batch4 should be marked manual probe only")
    _assert(not guard["leaderboard_eligible"], "manual batch4 must not be leaderboard eligible")


def test_oom_repair_batch8_allowed() -> None:
    guard = resource_feasibility_guard({
        "arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 8,
        "_config_repair_patch": {"batch_size": 8}, "repair_context": "OOM repair after CUDA out of memory",
    })
    _assert(not guard["resource_guard_blocked"], "OOM repair batch8 config should remain allowed")
    _assert(guard["oom_repair_context"], "OOM repair marker should be preserved in metadata")
    _assert(not guard["leaderboard_eligible"], "OOM repair batch8 must not be leaderboard eligible")


def test_oom_repair_batch4_blocked() -> None:
    guard = resource_feasibility_guard({
        "arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 4,
        "_config_repair_patch": {"batch_size": 4}, "repair_context": "OOM repair after CUDA out of memory",
    })
    _assert(guard["resource_guard_blocked"], "OOM repair batch4 should be blocked by automatic batch8 floor")


def test_normal_batch16_unaffected() -> None:
    guard = resource_feasibility_guard({"arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 16})
    _assert(not guard["resource_guard_triggered"], "normal batch16 proposal should be unaffected")
    _assert(guard["leaderboard_eligible"], "normal batch16 should be leaderboard eligible")


def test_cno_suggested_safe_config_excludes_sub8() -> None:
    guard = resource_feasibility_guard({"arch_name": "cno", "n_c": 44, "depth": 6, "batch_size": 16})
    options = guard["suggested_safe_config"]["batch_size_options"]
    _assert(options == [8], f"CNO safe config should only suggest batch8, got {options}")
    _assert(not any(b in options for b in (1, 2, 4)), f"safe config must not suggest sub8 batches: {options}")


def test_h100_probe_floor_oom_auto_fails() -> None:
    result = {
        "status": "failed",
        "arch_name": "plain_cnn",
        "config": {
            "arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 8,
            "resource_probe_only": True,
        },
        "log_tail": "2026 | INFO | Device=cuda GPU=NVIDIA H100 80GB HBM3 params=123\nRuntimeError: CUDA out of memory",
    }
    cls = classify_result(result)
    _assert(cls["classification"] == "AUTO_FAIL_RESOURCE_GUARD", f"unexpected classification: {cls}")
    _assert(cls["next_action"] == "AUTO_FAIL_RESOURCE_GUARD", "active H100 probe-floor OOM should not schedule RETRY")


def test_safe_config_patch_does_not_reduce_batch8_to4() -> None:
    patch = _safe_config_patch(
        {"config": {"arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 8}, "log_tail": "CUDA out of memory"},
        {"diagnosis": "OOM", "fix_description": "reduce batch"},
    )
    _assert("batch_size" not in patch, f"automatic OOM repair must not reduce batch8 to4: {patch}")


def test_batch8_h100_oom_auto_fails_not_reduced_to4() -> None:
    result = {
        "status": "failed",
        "arch_name": "plain_cnn",
        "config": {"arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 8, "_config_repair_patch": {"batch_size": 8}},
        "log_tail": "INFO | Device=cuda GPU=NVIDIA H100 80GB HBM3 params=123\nRuntimeError: CUDA out of memory",
    }
    cls = classify_result(result)
    _assert(cls["classification"] == "AUTO_FAIL_RESOURCE_GUARD", f"batch8 H100 repair OOM should auto-fail: {cls}")
    _assert(cls["next_action"] == "AUTO_FAIL_RESOURCE_GUARD", "batch8 H100 OOM must not schedule retry/reduce to4")


def test_h100_resume_l40s_oom_classified_gpu_downgrade_retry() -> None:
    result = {
        "status": "failed",
        "arch_name": "terrain_conditioned_local_attention_unet",
        "config": {"arch_name": "terrain_conditioned_local_attention_unet", "n_c": 20, "depth": 6, "batch_size": 16},
        "log_tail": "\n".join([
            'DeviceName = "NVIDIA H100 80GB HBM3"',
            "INFO | Device=cuda GPU=NVIDIA H100 80GB HBM3 params=6806618",
            "INFO | Batch 16 OK",
            "Job was evicted.",
            'DeviceName = "NVIDIA L40S"',
            "INFO | Device=cuda GPU=NVIDIA L40S params=6806618",
            "INFO | Resumed from epoch 150, best_val=0.097131",
            "ERROR | OOM even with batch=8 (CUDA out of memory); aborting this experiment.",
        ]),
    }
    cls = classify_result(result)
    _assert(cls["classification"] == "external_scheduler_EVICTED_GPU_DOWNGRADE", f"unexpected downgrade classification: {cls}")
    _assert(cls["next_action"] == "RETRY", f"GPU downgrade OOM should retry: {cls}")


def test_l40s_oom_batch16_without_prior_h100_retries() -> None:
    result = {
        "status": "failed",
        "arch_name": "plain_cnn",
        "config": {"arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 16},
        "log_tail": "INFO | Device=cuda GPU=NVIDIA L40S params=123\nRuntimeError: CUDA out of memory",
    }
    cls = classify_result(result)
    _assert(cls["next_action"] == "RETRY", f"ordinary L40S OOM should retry: {cls}")
    _assert(cls["classification"] in {"CUDA_OOM", "HIGH_VRAM"}, f"ordinary L40S OOM should not auto-fail: {cls}")


def test_transient_oom_still_retries() -> None:
    result = {
        "status": "failed",
        "arch_name": "plain_cnn",
        "config": {"arch_name": "plain_cnn", "n_c": 16, "depth": 4, "batch_size": 16},
        "log_tail": "RuntimeError: CUDA out of memory",
    }
    cls = classify_result(result)
    _assert(cls["next_action"] == "RETRY", f"ordinary OOM should stay retryable: {cls}")


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("OK: resource guard and deterministic OOM classifier tests passed locally.")


if __name__ == "__main__":
    main()






