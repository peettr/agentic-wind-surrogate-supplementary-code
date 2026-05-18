"""Prompt construction for Model Scout / Research Engine.

Supports multiple modes via ``modes.py``. The main entry point is
``build_prompt_for()`` which assembles a prompt for one specific AI
given the active mode and research context.
"""
from __future__ import annotations

import json
from typing import Any

from .modes import get_mode, MODES


# -----------------------------------------------------------------------
# Per-AI labels (used in prompt regardless of mode)
# -----------------------------------------------------------------------
AI_LABELS: dict[str, str] = {
    "glm": "GLM-5.1 (Zhipu)",
    "claude": "Claude Opus 4.7",
    "codex": "Codex GPT-5.4",
    "deepseek": "DeepSeek",
    "mimo": "MiMo Pro",
    "gemini": "Gemini",
    "grok": "Grok",
    "deep_research": "Google Deep Research",
}

# Backward compat: PROMPT_DIRECTIONS
PROMPT_DIRECTIONS = {name: {"label": label} for name, label in AI_LABELS.items()}


# -----------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------
def build_prompt_for(
    ai_name: str,
    *,
    mode: str = "model_scout",
    research_question: str = "",
    task_context: str = "",
    baseline: dict[str, Any] | None = None,
    existing_models: list[str] | None = None,
    n_proposals: int = 5,
    output_schema_override: str | None = None,
) -> str:
    """Assemble the prompt for one specific AI model.

    Args:
        ai_name:              key into ``AI_LABELS`` (e.g. ``"claude"``).
        mode:                 research mode (``"model_scout"``, ``"literature"``, etc.).
        research_question:    the core question (used in literature/sota/custom modes).
        task_context:         background context for the research.
        baseline:             dict describing the current baseline (model_scout mode).
        existing_models:      names already registered â€” the AI avoids these.
        n_proposals:          target number of results per AI.
        output_schema_override: custom output schema (overrides mode default).
    """
    if ai_name not in AI_LABELS:
        raise KeyError(f"unknown AI: {ai_name!r} (known: {list(AI_LABELS)})")

    mode_config = get_mode(mode)
    label = AI_LABELS[ai_name]
    existing = existing_models or []

    # --- Resolve direction ---
    direction = mode_config["per_ai_direction"] or ""
    if ai_name == "deep_research" and research_question:
        direction = (
            f"Conduct a thorough multi-hop research on: {research_question}. "
            "Search the web extensively, cross-reference multiple sources, "
            "and produce a comprehensive summary."
        )

    # --- Resolve task context ---
    ctx = task_context or mode_config.get("task_context") or ""
    if baseline and mode == "model_scout":
        ctx += f"\n\n### Baseline\n{json.dumps(baseline, indent=2)}"

    # --- Resolve output schema ---
    schema = output_schema_override or mode_config.get("output_schema") or ""
    if not schema:
        schema = 'Return your findings as a JSON object with a "proposals" key.'

    # --- Extra instructions ---
    extra = mode_config.get("extra_instructions", "")

    # --- Already-registered block ---
    already_block = ""
    if existing:
        already_block = (
            f"\n### Already found (skip these)\n{json.dumps(existing, indent=2)}"
        )

    # --- Assemble ---
    parts = [
        f"You are {label} acting as a research assistant.\n",
    ]
    if ctx:
        parts.append(ctx)
    if research_question and mode != "model_scout":
        parts.append(f"\n### Research Question\n{research_question}\n")
    parts.append(f"\n### Your search direction\n{direction}\n")
    parts.append(
        f"\nProduce ~{n_proposals} distinct entries. "
        "For each, provide enough detail for a follow-up agent to locate "
        "the source material.\n"
    )
    if extra:
        parts.append(f"\n{extra}\n")
    if already_block:
        parts.append(already_block)
    parts.append(f"\n{schema}")

    return "".join(parts)


# -----------------------------------------------------------------------
# Backward compat: original function signature
# -----------------------------------------------------------------------
def build_prompt_for_legacy(
    ai_name: str,
    problem: dict[str, Any],
    baseline: dict[str, Any],
    existing_models: list[str] | None = None,
    n_proposals: int = 5,
) -> str:
    """Original model-scout-only interface (backward compat)."""
    return build_prompt_for(
        ai_name,
        mode="model_scout",
        baseline=baseline,
        existing_models=existing_models,
        n_proposals=n_proposals,
    )


__all__ = [
    "build_prompt_for",
    "build_prompt_for_legacy",
    "PROMPT_DIRECTIONS",
    "AI_LABELS",
]



