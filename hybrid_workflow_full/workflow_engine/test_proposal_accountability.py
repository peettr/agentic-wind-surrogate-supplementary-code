from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from workflow_planner import (
    _flatten_proposal_item,
    build_proposal_rationale,
    extract_proposal_wrapper,
)


def test_flatten_merges_nested_and_top_level_rationale():
    item = {
        "experiment": {
            "experiment_id": "x1",
            "arch_name": "new_arch",
            "n_c": 16,
            "depth": 6,
            "loss_name": "masked_l1",
            "lr": 0.0005,
            "batch_size": 16,
            "input_features": "height",
            "epochs": 200,
            "seed": 1,
        },
        "role": "explorer",
        "_proposal_rationale": {
            "hypothesis": "nested hypothesis",
            "review_recommendation_addressed": "R3 rec",
            "deviation_reason": "testing a distinct mechanism",
        },
    }
    flat = _flatten_proposal_item(item)
    r = flat["_proposal_rationale"]
    assert r["role"] == "explorer"
    assert r["hypothesis"] == "nested hypothesis"
    assert r["review_recommendation_addressed"] == "R3 rec"
    assert r["deviation_reason"] == "testing a distinct mechanism"


def test_extract_wrapper_preserves_pack_metadata():
    text = '''{
      "weak_setting_budget_explanation": "only one weak control",
      "review_accountability_summary": "all proposals cite review",
      "proposals": [{
        "experiment_id": "x2", "arch_name": "arch", "n_c": 16, "depth": 6,
        "loss_name": "masked_l1", "lr": 0.0005, "batch_size": 16,
        "input_features": "height", "epochs": 200, "seed": 1
      }]
    }'''
    wrapper = extract_proposal_wrapper(text)
    assert wrapper["weak_setting_budget_explanation"] == "only one weak control"
    assert wrapper["review_accountability_summary"] == "all proposals cite review"
    assert len(wrapper["proposals"]) == 1


def test_rationale_debt_counts_missing_audit_fields():
    proposals = [{
        "experiment_id": "x3",
        "arch_name": "arch",
        "n_c": 16,
        "depth": 6,
        "loss_name": "masked_l1",
        "lr": 0.0005,
        "batch_size": 16,
        "input_features": "height",
        "epochs": 200,
        "seed": 1,
        "_proposal_rationale": {"role": "explorer"},
    }]
    report = build_proposal_rationale(proposals, 5, {"seen_arch_names": [], "seen_config_keys": []})
    debt = report["rationale_debt"]
    assert debt["total_warnings"] > 0
    assert debt["by_type"]["missing_review_accountability"] == 1
    assert debt["by_type"]["explorer_missing_mechanism"] == 1


if __name__ == "__main__":
    test_flatten_merges_nested_and_top_level_rationale()
    test_extract_wrapper_preserves_pack_metadata()
    test_rationale_debt_counts_missing_audit_fields()
    print("PASS proposal accountability tests")



