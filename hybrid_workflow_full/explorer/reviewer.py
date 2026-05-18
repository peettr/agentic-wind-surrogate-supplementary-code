"""V4 Result Reviewer â€” Analyzes experiment results between rounds.

Called after each round to:
1. Validate result quality (metrics completeness, anomalies)
2. Analyze trends (which models/HPs are improving)
3. Generate a review summary for the next AI prompt
4. Flag issues (code errors, regressions, stagnation)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

from explorer.explorer import ExperimentResult, summarize_results

LOGGER = logging.getLogger("auto_v6.reviewer")


# RÂ² thresholds based on V3 observations
R2_EXCELLENT = 0.70    # Top tier
R2_GOOD = 0.60         # Competitive
R2_WEAK = 0.40         # Below expectations
R2_FAILED = 0.20       # Likely code issue
R2_NEGATIVE = 0.0      # Broken


@dataclass
class RoundReview:
    """Structured review of a round's results."""
    round_num: int
    phase: str
    n_completed: int
    n_failed: int
    best_r2: float
    best_arch: str
    r2_vs_previous: float  # improvement over previous best
    issues: list[str]
    recommendations: list[str]
    model_ranking: list[dict]  # arch -> best R2, sorted
    stagnation_detected: bool
    summary: str


def review_round(
    history: list[ExperimentResult],
    round_num: int,
    phase: str,
    previous_best: float = float("-inf"),
) -> dict:
    """Review results from the latest round.

    Args:
        history: Full experiment history
        round_num: Current round number
        phase: Current phase (baseline/smoke/focus/long)
        previous_best: Best RÂ² from previous round

    Returns:
        RoundReview as dict
    """
    completed = [r for r in history if getattr(r, "status", "") == "ok"]
    failed = [r for r in history if getattr(r, "status", "") != "ok"]

    issues = []
    recommendations = []

    # --- Find best result ---
    best_r2 = float("-inf")
    best_arch = ""
    for r in completed:
        r2 = getattr(r, "val_r2_median", None)
        if r2 is not None and r2 > best_r2:
            best_r2 = r2
            best_arch = getattr(r, "arch_name", "unknown")

    if best_r2 == float("-inf"):
        best_r2 = float("nan")

    # --- Check for issues ---
    # High failure rate
    total = len(completed) + len(failed)
    if total > 0 and len(failed) / total > 0.5:
        issues.append(f"High failure rate: {len(failed)}/{total} failed")
        recommendations.append("Check code quality for recently added models")

    # Negative RÂ² (broken model)
    for r in completed:
        r2 = getattr(r, "val_r2_median", None)
        arch = getattr(r, "arch_name", "unknown")
        if r2 is not None and r2 < R2_NEGATIVE:
            issues.append(f"{arch} has negative R2 ({r2:.4f}) â€” likely broken")
            recommendations.append(f"Disable {arch} or fix implementation")

    # RÂ² regression
    r2_vs_previous = best_r2 - previous_best if previous_best != float("-inf") else 0.0
    if r2_vs_previous < -0.05:
        issues.append(f"R2 regression: {best_r2:.4f} vs previous {previous_best:.4f}")
        recommendations.append("Check if recent code changes introduced bugs")

    # Stagnation
    stagnation_detected = False
    if phase in ("focus", "long") and r2_vs_previous < 0.005 and round_num > 3:
        stagnation_detected = True
        issues.append(f"Stagnation: improvement only {r2_vs_previous:.4f}")
        recommendations.append("Consider new architectures or input features")

    # --- Model ranking ---
    model_best = {}
    for r in completed:
        arch = getattr(r, "arch_name", "unknown")
        r2 = getattr(r, "val_r2_median", None)
        if r2 is not None:
            if arch not in model_best or r2 > model_best[arch]:
                model_best[arch] = r2

    ranking = sorted(model_best.items(), key=lambda x: x[1], reverse=True)

    # Phase-specific recommendations
    if phase == "smoke":
        weak_models = [arch for arch, r2 in ranking if r2 < R2_WEAK]
        if weak_models:
            recommendations.append(f"Weak smoke results: {', '.join(weak_models)} â€” "
                                   "consider disabling or fixing")
        strong_models = [arch for arch, r2 in ranking if r2 >= R2_GOOD]
        if strong_models:
            recommendations.append(f"Promote to focus: {', '.join(strong_models)}")

    elif phase == "focus":
        top_models = [arch for arch, r2 in ranking[:3] if r2 >= R2_GOOD]
        if top_models:
            recommendations.append(f"Ready for long training: {', '.join(top_models)}")

    # --- Build summary ---
    summary_parts = [
        f"Round {round_num} ({phase}): {len(completed)} ok, {len(failed)} failed",
        f"Best: {best_arch} R2={best_r2:.4f}",
    ]
    if r2_vs_previous != 0:
        direction = "up" if r2_vs_previous > 0 else "down"
        summary_parts.append(f"Î”R2={r2_vs_previous:+.4f} ({direction})")
    if issues:
        summary_parts.append(f"Issues: {len(issues)}")
    summary = " | ".join(summary_parts)

    review = RoundReview(
        round_num=round_num,
        phase=phase,
        n_completed=len(completed),
        n_failed=len(failed),
        best_r2=best_r2 if best_r2 != float("nan") else None,
        best_arch=best_arch,
        r2_vs_previous=r2_vs_previous if r2_vs_previous != float("nan") else None,
        issues=issues,
        recommendations=recommendations,
        model_ranking=[{"arch": a, "best_r2": r} for a, r in ranking],
        stagnation_detected=stagnation_detected,
        summary=summary,
    )

    LOGGER.info(summary)
    for rec in recommendations:
        LOGGER.info("  -> %s", rec)

    return asdict(review)


def build_review_context(review: dict) -> str:
    """Build a text summary of the review to include in the next AI prompt."""
    lines = [
        f"## Previous Round Review (Round {review['round_num']}, {review['phase']})",
        f"Best: {review['best_arch']} R2={review['best_r2']:.4f}",
        f"Change: {review['r2_vs_previous']:+.4f}",
    ]
    if review["issues"]:
        lines.append("Issues:")
        for i in review["issues"]:
            lines.append(f"  - {i}")
    if review["recommendations"]:
        lines.append("Recommendations:")
        for r in review["recommendations"]:
            lines.append(f"  - {r}")
    if review["model_ranking"]:
        lines.append("Model ranking:")
        for entry in review["model_ranking"][:10]:
            lines.append(f"  {entry['arch']}: R2={entry['best_r2']:.4f}")
    return "\n".join(lines)

