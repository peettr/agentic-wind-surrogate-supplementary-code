import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.campaign_orchestrator import (
    build_benchmark_control_from_smoke_passes,
    build_smoke_control_from_candidates,
    is_submit_enabled,
    write_campaign_artifacts,
)


def _candidate(arch: str, source_run_id: str | None = None, *, loss="masked_l1", lr=0.001):
    rid = source_run_id or f"r_src_{arch}_hp"
    return {
        "source_campaign": "v5_ai_curated_001",
        "source_run_id": rid,
        "arch_name": arch,
        "model_file": f"generated_models/v5_codegen_batch_999/{arch}.py",
        "module_name": f"codegen_{arch}",
        "submit_tier": "a40_rtx6k_16gb",
        "batch_size": 16,
        "loss_name": loss,
        "lr": lr,
        "reason": f"candidate {arch} with {loss} lr={lr}",
    }


def test_build_smoke_control_selects_ten_new_hp_combinations_deterministically():
    candidates = [_candidate(f"arch_{i:02d}", loss="masked_l1_gradient" if i % 2 else "masked_l1", lr=0.0005 if i % 3 else 0.001) for i in range(14)]
    control = build_smoke_control_from_candidates(
        candidates,
        campaign="v5_controller_auto10_001_smoke20",
        run_prefix="r_auto10",
        count=10,
        exclude_arches={"arch_00", "arch_01"},
    )

    assert control["campaign"] == "v5_controller_auto10_001_smoke20"
    assert control["stage"] == "smoke20"
    assert len(control["runs"]) == 10
    assert all(row["source_campaign"] == "v5_ai_curated_001" for row in control["runs"])
    assert [row["arch_name"] for row in control["runs"]] == [f"arch_{i:02d}" for i in range(2, 12)]
    assert [row["run_id"] for row in control["runs"]] == [f"r_auto10_{i:02d}_arch_{i+2:02d}_smoke20" for i in range(10)]
    assert all("loss=" in row["reason"] and "lr=" in row["reason"] for row in control["runs"])


def test_write_campaign_artifacts_is_plan_only_by_default(tmp_path):
    candidates = [_candidate(f"arch_{i:02d}") for i in range(10)]
    control = build_smoke_control_from_candidates(
        candidates,
        campaign="v5_controller_auto10_001_smoke20",
        run_prefix="r_auto10",
        count=10,
    )
    outputs = write_campaign_artifacts(
        report_dir=tmp_path / "reports" / "auto10",
        smoke_control=control,
        materialize=False,
        submit=False,
        live_crc=False,
    )

    assert outputs["smoke_control"].exists()
    assert outputs["smoke_plan"].exists()
    assert not outputs.get("submitted_plan")
    saved_control = json.loads(outputs["smoke_control"].read_text())
    saved_plan = json.loads(outputs["smoke_plan"].read_text())
    assert saved_control["runs"][0]["run_id"].endswith("_smoke20")
    assert saved_plan["safety"] == {
        "materialize": False,
        "submit": False,
        "live_crc": False,
        "execute_repair": False,
    }


@pytest.mark.parametrize(
    "materialize,submit,dry_run,expected",
    [
        (False, False, True, False),
        (True, False, True, False),
        (True, True, True, False),
        (False, True, False, False),
        (True, True, False, True),
    ],
)
def test_submit_requires_materialize_and_not_dry_run(materialize, submit, dry_run, expected):
    assert is_submit_enabled(materialize=materialize, submit=submit, dry_run=dry_run) is expected


def test_promote_smoke_passes_to_benchmark200_control_only_when_all_passed():
    smoke_control = build_smoke_control_from_candidates(
        [_candidate(f"arch_{i:02d}") for i in range(3)],
        campaign="v5_controller_auto10_001_smoke20",
        run_prefix="r_auto10",
        count=3,
    )
    pass_state = {
        "runs": {
            row["run_id"]: {"state_key": "PASS:RECORD_RESULT", "current_run_id": row["run_id"]}
            for row in smoke_control["runs"]
        }
    }
    bench = build_benchmark_control_from_smoke_passes(
        smoke_control,
        pass_state,
        campaign="v5_controller_auto10_001_benchmark200",
        run_prefix="r_auto10b",
    )

    assert bench["stage"] == "benchmark200"
    assert bench["config_overrides"] == {"epochs": 200, "batch_size": 16, "strategy": "v5_benchmark200_codegen"}
    assert [row["source_run_id"] for row in bench["runs"]] == [row["run_id"] for row in smoke_control["runs"]]
    assert all(row["run_id"].endswith("_benchmark200") for row in bench["runs"])
    assert all("batch_size" not in row for row in bench["runs"])

    pass_state["runs"][smoke_control["runs"][2]["run_id"]] = {
        "state_key": "PASS:RECORD_RESULT",
        "current_run_id": smoke_control["runs"][2]["run_id"] + "_repair4",
        "metrics": {"val_metrics": {"r2_median": 0.01}},
        "promotion_allowed": False,
        "smoke_stage_policy": "code_validation_only",
    }
    bench_after_repair = build_benchmark_control_from_smoke_passes(
        smoke_control,
        pass_state,
        campaign="v5_controller_auto10_001_benchmark200",
        run_prefix="r_auto10b",
    )
    assert bench_after_repair["runs"][2]["source_run_id"] == smoke_control["runs"][2]["run_id"] + "_repair4"
    assert "performance" not in bench_after_repair["runs"][2]["reason"].lower()

    bad_state = {"runs": dict(pass_state["runs"])}
    bad_state["runs"][smoke_control["runs"][1]["run_id"]] = {"state_key": "RETRY:RETRY"}
    with pytest.raises(SystemExit, match="not all smoke runs passed"):
        build_benchmark_control_from_smoke_passes(
            smoke_control,
            bad_state,
            campaign="v5_controller_auto10_001_benchmark200",
            run_prefix="r_auto10b",
        )


def test_cli_plan_does_not_submit_or_touch_crc_by_default(tmp_path):
    out_dir = tmp_path / "auto10"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/campaign_orchestrator.py",
            "--output-dir",
            str(out_dir),
            "--count",
            "10",
            "--dry-run",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["mode"] == "plan-only"
    assert payload["safety"]["submit"] is False
    assert payload["safety"]["live_crc"] is False
    assert (out_dir / "control_smoke20.json").exists()
    assert (out_dir / "launch_plan_smoke20.json").exists()
