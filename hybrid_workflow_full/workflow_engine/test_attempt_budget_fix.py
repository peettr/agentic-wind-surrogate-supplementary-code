"""Verification tests for the attempt budget accounting fix.

Tests that:
1. PASS (completed with metrics) is never overridden by max attempts.
2. Failed retries still respect retry/total limits.
3. Existing retry completion does not double-count.
4. Duplicate PASS across rounds gets DUPLICATE_PASS classification but
   entry.status stays PASS.
5. PASS entry prevents check_limit from returning AUTO_FAIL_*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure we import from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attempt_manifest import (
    record_attempt,
    check_limit,
    ensure_run,
    load_manifest,
    base_run_id,
    TERMINAL_STATUSES,
)


def fresh_manifest() -> dict:
    return {"version": 1, "runs": {}}


def test_pass_not_overridden_by_max_total_attempts():
    """A completed-with-metrics result must stay PASS even when total >= max."""
    mf = fresh_manifest()
    base_id = "unet_v2_baseline_nc48_lr0.00025_masked_l1_gradient"

    # Record 4 failed attempts (OOM + retries) to reach total_attempts = 4
    for i in range(4):
        run_id = f"r029_smoke_{base_id}_retry{i}" if i > 0 else f"r029_smoke_{base_id}"
        atype = "retry" if i > 0 else "initial"
        record_attempt(
            mf, run_id, "smoke", atype, "failed", "CUDA_OOM", "RETRY",
            config={"arch_name": "unet_v2_baseline"},
        )

    entry = mf["runs"][base_id]
    assert entry["total_attempts"] == 4, f"expected 4, got {entry['total_attempts']}"

    # 5th attempt completes with metrics (PASS)
    run_id = f"r029_smoke_{base_id}_retry4"
    entry = record_attempt(
        mf, run_id, "smoke", "retry", "completed", "PASS", "PASS",
        config={"arch_name": "unet_v2_baseline"},
        metrics_path="/results/metrics.json",
    )

    assert entry["status"] == "PASS", f"Expected PASS, got {entry['status']}"
    assert entry["total_attempts"] == 5
    assert entry["metrics_path"] == "/results/metrics.json"

    # Even if check_limit is called explicitly for PASS, it must return None
    limit = check_limit(entry, "PASS", "retry", current_classification="PASS")
    assert limit is None, f"check_limit should return None for PASS, got {limit}"

    print("  PASS test_pass_not_overridden_by_max_total_attempts ✓")


def test_pass_not_overridden_by_check_limit_override():
    """Simulate the controller flow: classify_result→PASS, check_limit→should be None."""
    mf = fresh_manifest()
    base_id = "boundary_crossattn_unet_nc32_lr0.00025_masked_l1_gradient"

    # Pre-fill 4 attempts (smoke fail, retry pass, full fail, retry pending)
    record_attempt(mf, f"r025_smoke_{base_id}", "smoke", "initial", "failed", "CUDA_OOM", "RETRY", {})
    record_attempt(mf, f"r025_smoke_{base_id}_retry1", "smoke", "retry", "completed", "PASS", "PASS", {})
    record_attempt(mf, f"r025_full_{base_id}", "full", "initial", "failed", "CUDA_OOM", "RETRY", {})
    record_attempt(mf, f"r025_full_{base_id}_retry2", "full", "retry", "submitted", "CUDA_OOM", "RETRY", {})

    entry = mf["runs"][base_id]
    assert entry["total_attempts"] == 4
    assert entry["status"] == "PASS"  # First retry was PASS

    # Now the full retry completes with metrics
    # Controller flow: classify_result → PASS, then check_limit
    cls_classification = "PASS"
    action = "PASS"
    attempt_type = "retry"

    limit_status = check_limit(entry, action, attempt_type, current_classification=cls_classification)
    assert limit_status is None, f"check_limit must not override PASS, got {limit_status}"

    # Now record the attempt
    entry = record_attempt(
        mf, f"r025_full_{base_id}_retry2", "full", "retry", "completed",
        "PASS", "PASS", {},
        metrics_path="/results/metrics.json",
    )
    assert entry["status"] == "PASS", f"Expected PASS, got {entry['status']}"

    print("  PASS test_pass_not_overridden_by_check_limit_override ✓")


def test_failed_retry_respects_limits():
    """Failed retries should still be caught by limit checks."""
    mf = fresh_manifest()
    base_id = "some_model_nc24_lr0.0003"

    # 3 retries exhausted
    record_attempt(mf, f"r025_smoke_{base_id}", "smoke", "initial", "failed", "CUDA_OOM", "RETRY", {})
    record_attempt(mf, f"r025_smoke_{base_id}_retry1", "smoke", "retry", "failed", "CUDA_OOM", "RETRY", {})
    record_attempt(mf, f"r025_smoke_{base_id}_retry2", "smoke", "retry", "failed", "CUDA_OOM", "RETRY", {})

    entry = mf["runs"][base_id]
    assert entry["retry_count"] == 2
    assert entry["total_attempts"] == 3

    # Next retry should be blocked by max_retries
    limit = check_limit(entry, "RETRY", "retry")
    assert limit == "AUTO_FAIL_MAX_RETRIES", f"Expected AUTO_FAIL_MAX_RETRIES, got {limit}"

    # And total limit with 5 attempts
    record_attempt(mf, f"r025_smoke_{base_id}_retry3", "smoke", "retry", "failed", "CUDA_OOM", "RETRY", {})
    record_attempt(mf, f"r025_full_{base_id}", "full", "initial", "failed", "CUDA_OOM", "RETRY", {})
    entry = mf["runs"][base_id]
    assert entry["total_attempts"] == 5
    limit = check_limit(entry, "RETRY", "retry")
    assert limit == "AUTO_FAIL_MAX_TOTAL_ATTEMPTS", f"Expected AUTO_FAIL_MAX_TOTAL_ATTEMPTS, got {limit}"

    print("  PASS test_failed_retry_respects_limits ✓")


def test_no_double_count_on_re_record():
    """Re-recording an existing run_id should update, not append."""
    mf = fresh_manifest()
    base_id = "some_model_nc24_lr0.0003"

    # Record initial
    record_attempt(mf, f"r025_smoke_{base_id}", "smoke", "initial", "submitted", "IN_PROGRESS", "WAIT", {})
    entry = mf["runs"][base_id]
    assert entry["total_attempts"] == 1
    assert entry["retry_count"] == 0

    # Re-record with updated status (simulating monitor update)
    record_attempt(mf, f"r025_smoke_{base_id}", "smoke", "initial", "completed", "PASS", "PASS", {},
                   metrics_path="/results/metrics.json")
    entry = mf["runs"][base_id]
    assert entry["total_attempts"] == 1, f"Should not double-count: {entry['total_attempts']}"
    assert entry["status"] == "PASS"
    assert len(entry["attempts"]) == 1

    print("  PASS test_no_double_count_on_re_record ✓")


def test_duplicate_pass_across_rounds():
    """A config that PASS'd in R25 full should not get AUTO_FAIL when re-proposed in R27."""
    mf = fresh_manifest()
    base_id = "boundary_crossattn_unet_nc24_lr0.0003_masked_l1_gradient"

    # R25: smoke PASS + full PASS
    record_attempt(mf, f"r025_smoke_{base_id}", "smoke", "initial", "completed", "PASS", "PASS", {})
    record_attempt(mf, f"r025_full_{base_id}", "full", "initial", "completed", "PASS", "PASS", {})

    entry = mf["runs"][base_id]
    assert entry["status"] == "PASS"
    assert entry["total_attempts"] == 2

    # R26: same config re-proposed, smoke PASS
    limit = check_limit(entry, "PASS", "initial", current_classification="PASS")
    assert limit is None, f"Already-PASS entry should not fail limit: {limit}"

    entry = record_attempt(mf, f"r026_smoke_{base_id}", "smoke", "initial", "completed", "PASS", "PASS", {})
    assert entry["status"] == "PASS", f"Should remain PASS: {entry['status']}"

    # Check the attempt was tagged as DUPLICATE_PASS
    dup_attempt = entry["attempts"][-1]
    assert dup_attempt.get("classification") == "DUPLICATE_PASS", f"Expected DUPLICATE_PASS: {dup_attempt.get('classification')}"

    # R27 full: also PASS, still no AUTO_FAIL
    entry = record_attempt(mf, f"r027_full_{base_id}", "full", "initial", "completed", "PASS", "PASS", {})
    assert entry["status"] == "PASS"

    print("  PASS test_duplicate_pass_across_rounds ✓")


def test_auto_fail_prevented_on_pass_entry():
    """Once an entry is PASS, record_attempt with AUTO_FAIL action must not overwrite."""
    mf = fresh_manifest()
    base_id = "test_model_nc24"

    # First: PASS
    record_attempt(mf, f"r025_smoke_{base_id}", "smoke", "initial", "completed", "PASS", "PASS", {})
    entry = mf["runs"][base_id]
    assert entry["status"] == "PASS"

    # Now try to record AUTO_FAIL_MAX_TOTAL_ATTEMPTS (simulating the bug)
    entry = record_attempt(
        mf, f"r026_smoke_{base_id}", "smoke", "initial", "completed",
        "AUTO_FAIL_MAX_TOTAL_ATTEMPTS", "AUTO_FAIL_MAX_TOTAL_ATTEMPTS", {},
    )
    assert entry["status"] == "PASS", f"PASS must not be overwritten: {entry['status']}"

    print("  PASS test_auto_fail_prevented_on_pass_entry ✓")


def test_real_manifest_boundary_crossattn_nc32():
    """Test against the actual Phase9 manifest data for boundary_crossattn_unet_nc32."""
    mf = fresh_manifest()
    base_id = "boundary_crossattn_unet_nc32_lr0.00025_masked_l1_gradient"

    # Reproduce the exact sequence from the manifest:
    # 0: r025_smoke, failed, CUDA_OOM → RETRY
    record_attempt(mf, f"r025_smoke_{base_id}", "smoke", "initial", "failed", "CUDA_OOM", "RETRY", {})
    # 1: r025_smoke_retry1, completed → PASS (smoke passed)
    entry = record_attempt(mf, f"r025_smoke_{base_id}_retry1", "smoke", "retry", "completed", "PASS", "PASS", {})
    assert entry["status"] == "PASS", "Smoke retry should be PASS"

    # 2: r025_full, failed, CUDA_OOM → RETRY
    record_attempt(mf, f"r025_full_{base_id}", "full", "initial", "failed", "CUDA_OOM", "RETRY", {})
    assert entry["status"] == "PASS", "Full fail should not override PASS"

    # 3: r025_full_retry2, completed with metrics → should be PASS not AUTO_FAIL
    limit = check_limit(entry, "PASS", "retry", current_classification="PASS")
    assert limit is None, f"Full retry pass should not trigger limit: {limit}"

    entry = record_attempt(mf, f"r025_full_{base_id}_retry2", "full", "retry", "completed", "PASS", "PASS", {},
                           metrics_path="/some/metrics.json")
    assert entry["status"] == "PASS", f"Full retry with metrics must be PASS: {entry['status']}"

    print("  PASS test_real_manifest_boundary_crossattn_nc32 ✓")


if __name__ == "__main__":
    print("Running attempt budget fix verification tests...\n")
    tests = [
        test_pass_not_overridden_by_max_total_attempts,
        test_pass_not_overridden_by_check_limit_override,
        test_failed_retry_respects_limits,
        test_no_double_count_on_re_record,
        test_duplicate_pass_across_rounds,
        test_auto_fail_prevented_on_pass_entry,
        test_real_manifest_boundary_crossattn_nc32,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failures += 1

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed.")
    if failures:
        sys.exit(1)
    else:
        print("All tests passed ✓")
