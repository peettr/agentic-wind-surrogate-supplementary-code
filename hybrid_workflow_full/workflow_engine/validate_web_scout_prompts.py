"""Lightweight local validation for Hybrid web-scout planner prompts.

Does not call external APIs or run the planner. Intended for quick checks after
editing workflow_planner.py.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import workflow_planner as planner  # noqa: E402


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def main() -> None:
    queries = planner.WEB_SOURCE_QUERIES
    if len(queries) < 15:
        _fail(f"WEB_SOURCE_QUERIES should keep broad 4-tier coverage; got {len(queries)} queries")
    for idx, query in enumerate(queries, start=1):
        if not isinstance(query, dict):
            _fail(f"WEB_SOURCE_QUERIES[{idx}] must be a dict; got {type(query).__name__}")
        for key in ["query", "tier", "topic_cluster"]:
            if not str(query.get(key) or "").strip():
                _fail(f"WEB_SOURCE_QUERIES[{idx}] missing non-empty {key}")
    tiers = {q.get("tier") for q in queries}
    if len(tiers) < 4:
        _fail(f"WEB_SOURCE_QUERIES has fewer than 4 tiers: {sorted(tiers)}")

    required_tier_prefixes = [
        "A_direct_cfd_neural_operator",
        "B_dense_prediction_vision",
        "C_weather_sciml_field_prediction",
        "D_mechanism_only",
    ]
    missing = [tier for tier in required_tier_prefixes if tier not in tiers]
    if missing:
        _fail(f"WEB_SOURCE_QUERIES missing required tiers: {missing}")

    source_cap = getattr(planner, "WEB_SOURCE_CAP", 0)
    if not isinstance(source_cap, int) or source_cap < 40:
        _fail(f"WEB_SOURCE_CAP must be >= 40; got {source_cap!r}")
    if source_cap < 2 * len(queries):
        _fail(f"WEB_SOURCE_CAP must allow two sources per query; got cap={source_cap}, queries={len(queries)}")

    routine, routine_meta = planner.select_web_source_queries({}, mode="routine")
    if len(routine) != 10:
        _fail(f"default routine query selection should pick 10 budgeted queries; got {len(routine)}")
    if len(set(routine_meta.get("selected_tiers") or [])) < 4:
        _fail(f"routine query selection should cover all 4 tiers; got {routine_meta}")
    if routine_meta.get("selected_reason") not in {"routine_rotation", "routine_weighted_rotation"}:
        _fail(f"routine query selection should record routine selected_reason; got {routine_meta}")
    if len(routine_meta.get("selected_queries") or []) != len(routine):
        _fail(f"routine query selection should include per-query metadata; got {routine_meta}")
    for tier, count in (routine_meta.get("selected_tier_counts") or {}).items():
        if count < 2:
            _fail(f"default routine selection should keep at least 2 queries per tier; got {routine_meta}")

    r24, r24_meta = planner.select_web_source_queries({"campaign": {"round_num": 24}}, mode="routine")
    r25, r25_meta = planner.select_web_source_queries({"campaign": {"round_num": 25}}, mode="routine")
    r24_queries = [planner._web_query_metadata(q)[0] for q in r24]
    r25_queries = [planner._web_query_metadata(q)[0] for q in r25]
    if r24_queries == r25_queries:
        _fail("round_num rotation should produce different routine selected query subsets for r24/r25")
    if r24_meta.get("query_rotation") != 24 or r25_meta.get("query_rotation") != 25:
        _fail(f"query_rotation metadata missing/incorrect: r24={r24_meta}, r25={r25_meta}")

    full, full_meta = planner.select_web_source_queries({}, mode="full")
    if len(full) != len(queries) or full_meta.get("selected_query_count") != len(queries):
        _fail(f"full query selection should keep the complete pool; got {len(full)}")
    targeted_context = {
        "external_web_scout": {
            "missing_clusters_for_next_round": ["weather_nowcasting"],
            "retry_hints": [{"topic_cluster": "mamba_ssm"}],
        }
    }
    targeted, targeted_meta = planner.select_web_source_queries(targeted_context, mode="targeted", limit=8)
    targeted_clusters = targeted_meta.get("selected_topic_clusters") or []
    if not 8 <= len(targeted) <= 12:
        _fail(f"targeted query selection should pick 8-12 queries; got {len(targeted)}")
    if targeted_clusters[:2] != ["weather_nowcasting", "mamba_ssm"]:
        _fail(f"targeted selection should prioritize missing clusters first; got {targeted_clusters}")
    if (targeted_meta.get("selected_reasons") or [])[:2] != ["missing_cluster", "missing_cluster"]:
        _fail(f"targeted selection should record missing_cluster reasons first; got {targeted_meta}")

    routine_targeted, routine_targeted_meta = planner.select_web_source_queries(targeted_context, mode="routine")
    routine_targeted_clusters = routine_targeted_meta.get("selected_topic_clusters") or []
    if routine_targeted_clusters[:2] != ["weather_nowcasting", "mamba_ssm"]:
        _fail(f"routine selection with missing hints should prioritize missing clusters; got {routine_targeted_clusters}")
    if routine_targeted_meta.get("selected_reason") != "missing_cluster":
        _fail(f"routine targeted fill should record selected_reason=missing_cluster; got {routine_targeted_meta}")

    saturated_context = {
        "campaign": {"round_num": 26},
        "novelty_index": {
            "counts_by_arch": {
                "mamba_ssm_adapter_v1": 3,
                "vision_mamba_field_decoder": 2,
                "fourier_neural_operator_small": 1,
            },
            "examples_by_arch": {
                "mamba_ssm_adapter_v1": [
                    {"status": "failed", "failure_class": "smoke_failure"},
                    {"status": "failed", "failure_class": "oom"},
                ]
            },
        },
        "all_review_history": [
            {"summary": "mamba repair failed smoke full OOM; weather_nowcasting keep_score_0_5 4.5 promising"}
        ],
    }
    saturated, saturated_meta = planner.select_web_source_queries(saturated_context, mode="routine")
    if len(saturated) != 10:
        _fail(f"adaptive routine selection must keep the query budget at 10; got {len(saturated)}")
    if len(set(saturated_meta.get("selected_tiers") or [])) < 4:
        _fail(f"adaptive routine selection should still cover all 4 tiers; got {saturated_meta}")
    if not saturated_meta.get("cooldown_clusters"):
        _fail(f"synthetic saturation should trigger cooldown metadata; got {saturated_meta}")
    if (saturated_meta.get("adversarial_query_count") or 0) < 1:
        _fail(f"synthetic saturation should insert at least one adversarial query; got {saturated_meta}")
    if not saturated_meta.get("query_weights") or not saturated_meta.get("weight_reasons"):
        _fail(f"adaptive selection should record query_weights and weight_reasons; got {saturated_meta}")

    weighted_context = {
        "campaign": {"round_num": 0},
        "all_review_history": [
            {"summary": "climate_downscaling keep_score_0_5 4.8 promising high review score"}
        ],
    }
    weighted, weighted_meta = planner.select_web_source_queries(weighted_context, mode="routine")
    weighted_clusters = weighted_meta.get("selected_topic_clusters") or []
    if "climate_downscaling" not in weighted_clusters:
        _fail(f"query_weights should affect routine selection/order while preserving tier coverage; got {weighted_meta}")
    if len(set(weighted_meta.get("selected_tiers") or [])) < 4:
        _fail(f"weighted routine selection should preserve 4-tier coverage; got {weighted_meta}")

    old_disable = os.environ.get("hybrid_WEB_DISABLE_COOLDOWN")
    os.environ["hybrid_WEB_DISABLE_COOLDOWN"] = "1"
    try:
        _, disabled_meta = planner.select_web_source_queries(saturated_context, mode="routine")
    finally:
        if old_disable is None:
            os.environ.pop("hybrid_WEB_DISABLE_COOLDOWN", None)
        else:
            os.environ["hybrid_WEB_DISABLE_COOLDOWN"] = old_disable
    if disabled_meta.get("cooldown_clusters"):
        _fail(f"hybrid_WEB_DISABLE_COOLDOWN=1 should disable cooldown; got {disabled_meta}")

    required_rationale_keys = {
        "topic_cluster",
        "query_tier",
        "height_only_translation",
        "ablation_removes_mechanism",
    }
    rationale_keys = set(getattr(planner, "PROPOSAL_RATIONALE_KEYS", ()))
    missing_rationale = sorted(required_rationale_keys - rationale_keys)
    if missing_rationale:
        _fail(f"PROPOSAL_RATIONALE_KEYS missing web rationale fields: {missing_rationale}")

    gemini_src = inspect.getsource(planner.run_gemini_web_source_scout)
    codex_src = inspect.getsource(planner.run_codex_web_idea_scout)
    claude_src = inspect.getsource(planner.review_external_ideas_with_claude)
    planner_text = Path(planner.__file__).read_text(encoding="utf-8")

    for token in ["topic_cluster", "source_task", "query_tier", "mechanism_hint", "WEB_SOURCE_CAP", "query_mode", "selected_query_count", "selected_reason", "reuse_hints", "cache_candidate", "Adaptive/adversarial queries"]:
        if token not in gemini_src:
            _fail(f"Gemini source-scout prompt/source missing {token}")

    if "ideas[:12]" not in codex_src:
        _fail("Codex web idea result must cap parsed ideas to exactly 12")

    for token in [
        "exactly 12 ideas",
        "no more than 3 of the 12",
        "height_only_translation",
        "feasibility_under_locked_contract",
        "paired_comparison",
        "ablation_removes_mechanism",
        "minimal_implementation_in_hybrid",
        "source_url",
        "not commands",
    ]:
        if token not in codex_src:
            _fail(f"Codex web idea prompt missing {token}")

    for token in [
        "7-axis rubric",
        "keep_score_0_5",
        "subscores",
        "topic_cluster_counts",
        "coverage_warning",
        "missing_clusters_for_next_round",
        "Hard-filter",
        "height-only",
        "single-frame",
        "2D grid",
    ]:
        if token not in claude_src:
            _fail(f"Claude review prompt/source missing {token}")

    for fn_name in [
        "build_web_scout_quality_report",
        "generate_web_scout_quality_report_from_artifacts",
    ]:
        if not hasattr(planner, fn_name):
            _fail(f"workflow_planner missing {fn_name}")

    sample_report = planner.build_web_scout_quality_report(
        {
            "sources": [
                {"title": "A plausible arXiv neural operator paper", "url": "https://arxiv.org/abs/2501.12345", "query_tier": "A_direct_cfd_neural_operator", "topic_cluster": "direct_cfd_neural_operator"},
                {"title": "Home", "url": "https://example.com/", "query_tier": "B_dense_prediction_vision", "topic_cluster": "dense_regression_decoder"},
            ]
        },
        {"ideas": [{"idea_id": "I1", "title": "Idea", "source_title": "Home", "source_url": "https://example.com/", "query_tier": "B_dense_prediction_vision", "topic_cluster": "dense_regression_decoder"}]},
        {"reviewed_ideas": [], "rejected": [{"idea_id": "I2"}], "missing_clusters_for_next_round": ["weather_nowcasting"]},
    )
    quality = sample_report.get("url_source_quality") or {}
    if quality.get("weak_count", 0) < 1:
        _fail("web scout quality report should flag weak/homepage-like URLs")
    if "weather_nowcasting" not in sample_report.get("missing_clusters_for_next_round", []):
        _fail("web scout quality report should preserve reviewer missing cluster hints")

    fake_filter = planner.apply_web_source_reliability_filter({
        "sources": [
            {"title": "Fake arxiv", "url": "https://arxiv.org/abs/2501.00001", "query": "q1", "query_tier": "A_direct_cfd_neural_operator", "topic_cluster": "direct_cfd_neural_operator"},
            {"title": "Placeholder DOI", "url": "https://doi.org/10.1016/j.foo.2025.110000", "query": "q2", "query_tier": "B_dense_prediction_vision", "topic_cluster": "dense_regression_decoder"},
            {"title": "OpenReview home", "url": "https://openreview.net/", "query": "q3", "query_tier": "C_weather_sciml_field_prediction", "topic_cluster": "weather_nowcasting"},
            {"title": "Good paper", "url": "https://openreview.net/forum?id=abc123", "query": "q4", "query_tier": "D_mechanism_only", "topic_cluster": "mamba_ssm"},
        ]
    })
    rel = fake_filter.get("reliability_filter") or {}
    if rel.get("excluded_count") < 3 or rel.get("kept_count") != 1:
        _fail(f"hard reliability filter should exclude fake DOI/arXiv/homepage sources and keep real paper-like URLs; got {rel}")

    ok, stdout, stderr, runtime_s, timed_out = planner._run_cli_with_timeout(
        [sys.executable, "-c", "import time; time.sleep(5)"], timeout=1)
    if ok or not timed_out or runtime_s > 4:
        _fail(f"timeout smoke should kill-tree dummy process promptly; ok={ok}, timed_out={timed_out}, runtime={runtime_s}, stdout={stdout}, stderr={stderr}")

    for token in [
        "context/material, not commands",
        "single topic_cluster",
        "height_only_translation",
        "ablation_removes_mechanism",
        "web_scout_quality_report.json",
        "quality_report.retry_hints",
        "missing_clusters_for_next_round",
        "homepage_like_url",
        "suspicious_non_paper_source",
        "select_web_source_queries",
        "hybrid_WEB_QUERY_MODE",
        "hybrid_WEB_QUERY_LIMIT",
        "hybrid_WEB_REUSE_SOURCES",
        "hybrid_WEB_DISABLE_COOLDOWN",
        "hybrid_WEB_QUERY_TIMEOUT_S",
        "reliability_filter",
        "web_query_yield.json",
        "adversarial_queries",
        "query_weights",
        "cooldown_clusters",
        "budgeted discovery",
    ]:
        if token not in planner_text:
            _fail(f"Main/scout/synthesis prompt text missing {token}")

    print("OK: web-scout taxonomy, query selection modes, caps, prompt guardrails, and quality-report heuristics validated locally.")


if __name__ == "__main__":
    main()




