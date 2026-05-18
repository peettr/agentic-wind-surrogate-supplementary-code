from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "workflow_engine"
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ROOT))

import workflow_codegen as cg
from workflow_codegen import normalize_future_imports, validate_model_code
from workflow_reviewer import run_post_codegen_validation
from schema_guards import validate_experiment_schema, validate_registered_arch_config


def _write_model(name: str, code: str) -> Path:
    cg.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = cg.GENERATED_DIR / f"{name}.py"
    path.write_text(code, encoding="utf-8")
    return path


def test_three_channel_hardcoded_conv2d_fails():
    name = "zz_test_hardcoded_conv"
    path = _write_model(name, """
import torch
import torch.nn as nn
class zz_test_hardcoded_conv(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=8, depth=3):
        super().__init__()
        self.c = nn.Conv2d(1, out_channels, 3, padding=1)
    def forward(self, x):
        return self.c(x[:, :1])
""")
    try:
        ok, msg = validate_model_code(name, {"input_features": "height_sdf_normal", "n_c": 8, "depth": 3})
        assert not ok and "INPUT_CHANNEL_MISMATCH" in msg
    finally:
        path.unlink(missing_ok=True)


def test_three_channel_variable_in_channels_passes():
    name = "zz_test_variable_conv"
    path = _write_model(name, """
import torch
import torch.nn as nn
class zz_test_variable_conv(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=8, depth=3):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    def forward(self, x):
        return self.c(x)
""")
    try:
        ok, msg = validate_model_code(name, {"input_features": "height_sdf_normal", "n_c": 8, "depth": 3})
        assert ok, msg
    finally:
        path.unlink(missing_ok=True)


def test_masked_l1_grad_rejected_not_normalized():
    cfg = {"arch_name": "unet_sdf_7level", "n_c": 16, "depth": 7, "loss_name": "masked_l1_grad", "lr": 5e-4, "batch_size": 16, "input_features": "height", "seed": 1}
    ok, issues = validate_experiment_schema(cfg, stage="planner")
    assert not ok
    assert any(i["code"] == "LOSS_REGISTRY_KEYERROR" for i in issues)
    assert cfg["loss_name"] == "masked_l1_grad"


def test_future_import_hoist_compiles():
    code = '"""doc"""\nimport torch\n\ndef helper():\n    return 1\nfrom __future__ import annotations\nimport torch.nn as nn\n'
    fixed, actions = normalize_future_imports(code)
    assert actions
    assert fixed.splitlines()[1] == "from __future__ import annotations"
    compile(fixed, "future_test.py", "exec")


def test_manifest_terminal_failure_makes_review_not_ok():
    name = "zz_test_review_ok"
    path = _write_model(name, """
import torch
import torch.nn as nn
class zz_test_review_ok(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_c=8, depth=3):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    def forward(self, x):
        return self.c(x)
""")
    try:
        state = {"proposals": [{"arch_name": name, "n_c": 8, "depth": 3, "input_features": "height"}]}
        manifest = {"generated_archs": [name], "validated_archs": [name], "failed_skipped_after_codegen_retries": ["bad_arch: generation failed"]}
        result = run_post_codegen_validation(state, manifest, [name])
        assert not result["ok"]
        assert result["manifest_failures"]
    finally:
        path.unlink(missing_ok=True)


def test_quadmamba_depth4_registered_blocked():
    ok, msg = validate_registered_arch_config("quadmamba", {"arch_name": "quadmamba", "n_c": 24, "depth": 4, "loss_name": "masked_l1", "lr": 3e-4, "batch_size": 16, "input_features": "height"})
    assert not ok and "ARCH_CONSTRAINT_FAIL" in msg


def test_r0_schema_dry_run_no_false_failures():
    runs = ROOT / "campaigns" / "auto_v6" / "runs"
    checked = 0
    failures = []
    for cfg_path in runs.glob("r000_smoke_*/*train_config.json"):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        compact = {
            "arch_name": cfg.get("arch_name"),
            "n_c": (cfg.get("arch_kwargs") or {}).get("n_c", 16),
            "depth": (cfg.get("arch_kwargs") or {}).get("depth", 7),
            "loss_name": cfg.get("loss_name"),
            "lr": cfg.get("lr"),
            "batch_size": cfg.get("batch_size"),
            "input_features": cfg.get("input_features"),
            "seed": cfg.get("seed"),
        }
        ok, issues = validate_experiment_schema(compact, stage="r0_dry_run")
        checked += 1
        if not ok:
            failures.append({"path": str(cfg_path), "issues": issues})
    assert checked > 0
    assert not failures, failures


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
