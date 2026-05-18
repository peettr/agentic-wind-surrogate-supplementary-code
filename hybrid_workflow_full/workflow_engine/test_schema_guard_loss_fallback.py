from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema_guards


def test_legal_loss_names_from_source_contains_locked_losses():
    names = schema_guards.legal_loss_names_from_source()
    assert {"masked_l1", "masked_l1_gradient", "masked_huber"}.issubset(names)


def test_validate_experiment_schema_accepts_masked_l1_gradient_without_torch_import():
    ok, issues = schema_guards.validate_experiment_schema({
        "arch_name": "dummy_arch",
        "loss_name": "masked_l1_gradient",
        "input_features": "height_sdf_normal",
        "n_c": 24,
        "depth": 6,
        "batch_size": 16,
        "lr": 0.0005,
    })
    assert ok, issues


if __name__ == "__main__":
    test_legal_loss_names_from_source_contains_locked_losses()
    test_validate_experiment_schema_accepts_masked_l1_gradient_without_torch_import()
    print("PASS schema guard loss fallback tests")



