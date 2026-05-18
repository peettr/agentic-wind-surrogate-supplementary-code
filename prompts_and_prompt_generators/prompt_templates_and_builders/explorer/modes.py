"""Research modes — preset configurations for different research tasks.

Each mode defines:
- prompt_template: how to build the per-AI prompt
- output_schema: what JSON structure AIs must return
- scoring_weights: how ProposalCollector scores results
- claude_summary_instructions: what Claude Opus should do in the summary step
"""
from __future__ import annotations

from typing import Any


# =====================================================================
# Mode: model_scout — neural architecture search
# =====================================================================
MODEL_SCOUT: dict[str, Any] = {
    "name": "model_scout",
    "label": "Model Architecture Scout",
    "description": "Search for neural network architectures suitable for a specific task.",
    "output_schema": """### Output format (MUST be a parseable JSON object)
```json
{
  "proposals": [
    {
      "name": "snake_case_name",
      "category": "cnn_encoder_decoder|transformer|operator|hybrid|inr|mlp_mixer|other",
      "variant_of": "unet|fno|null",
      "source": "paper title + venue + year, or 'AI-generated novel architecture'",
      "url": "https://arxiv.org/abs/XXXX.XXXXX or repo URL, null if none",
      "rationale": "1-2 sentences: why this architecture is well-suited",
      "novelty": "existing|generated",
      "estimated_params_m": 40,
      "estimated_vram_gb": 20,
      "difficulty": "easy|medium|hard"
    }
  ]
}
```
Return ONLY the JSON object, no prose outside it.""",

    "per_ai_direction": (
        "Search extensively for neural network architectures suitable for "
        "2D spatial field prediction (640x640 regression). Use web search "
        "to find the latest papers, repos, and techniques from any source: "
        "arxiv, conferences, GitHub, blogs. For each architecture explain "
        "why it could outperform a 7-level UNet baseline on urban wind "
        "field prediction. Prioritise architectures with reproducible code "
        "and published benchmarks."
    ),

    "task_context": (
        "Task: 2D spatial field prediction — regression from a building-height map to a "
        "wind-speed field. Single GPU inference, urban microclimate scale.\n\n"
        "Input  : tensor of shape (B, 1, 640, 640), float.\n"
        "Output : tensor of shape (B, 1, 640, 640), float, non-negative (final ReLU).\n"
        "Target contains NaN over building interiors — loss must be NaN-safe.\n"
        "Single-GPU VRAM budget: 80 GB. Batch size 16."
    ),

    "extra_instructions": (
        "IMPORTANT: Do not limit yourself to UNet variants. Paradigm-shifting "
        "architectures (e.g. neural operators, implicit representations, graph "
        "networks, diffusion-based models) are welcome even if unproven on this "
        "exact task — we want to explore, not just incrementally improve the baseline."
    ),

    "claude_summary_prompt": (
        "You are Claude Opus acting as a senior research advisor for neural network "
        "architecture search.\n\n"
        "Analyze the collected proposals, identify consensus, assess feasibility, "
        "re-rank by potential impact, optionally synthesize new architectures, "
        "and recommend a Top-5 shortlist with rationale and risk assessment.\n\n"
        "RULES: Keep ALL original proposals. You may add new synthesized ones. "
        "You may re-order. Output a well-structured Markdown report."
    ),

    "scoring": {
        "multi_scout_bonus": 3,     # proposed by 2+ AIs
        "has_arxiv_or_doi": 2,       # has academic citation
        "small_params": 1,           # < 50M params
        "low_vram": 1,              # < 40 GB VRAM
        "new_category": 2,          # different paradigm than existing
        "novelty_bonus": 1,         # AI-generated novel architecture
    },
}


# =====================================================================
# Mode: literature — literature review on a topic
# =====================================================================
LITERATURE: dict[str, Any] = {
    "name": "literature",
    "label": "Literature Review",
    "description": "Multi-AI literature search and synthesis on a research topic.",
    "output_schema": """### Output format (MUST be a parseable JSON object)
```json
{
  "proposals": [
    {
      "name": "paper_short_title",
      "category": "experimental|theoretical|computational|review|dataset",
      "source": "full paper title, all authors, venue/journal, year",
      "url": "arxiv DOI or publisher URL, null if none",
      "rationale": "2-3 sentences: key contribution and why it is relevant",
      "year": 2025,
      "methodology": "brief description of method",
      "key_finding": "main result in one sentence",
      "relevance": "high|medium|low"
    }
  ]
}
```
Return ONLY the JSON object, no prose outside it.""",

    "per_ai_direction": None,  # filled dynamically from research_question

    "task_context": None,  # filled dynamically from research_question

    "extra_instructions": (
        "For each paper, provide the FULL citation (all authors, title, venue, year). "
        "If you cannot verify a citation, say so explicitly. Prefer papers with "
        "reproducible results or open-source code. Include both seminal/foundational "
        "works and the latest (2024-2026) developments."
    ),

    "claude_summary_prompt": (
        "You are Claude Opus acting as a senior research advisor conducting a "
        "literature review.\n\n"
        "Analyze the collected papers, identify key themes and research threads, "
        "group papers by approach, highlight consensus and contradictions, "
        "identify gaps in the literature, and recommend the 5 most important "
        "papers to read first.\n\n"
        "RULES: Keep ALL original papers. You may add missed papers. "
        "You may re-order. Output a well-structured Markdown literature review."
    ),

    "scoring": {
        "multi_scout_bonus": 3,
        "has_arxiv_or_doi": 2,
        "recent_year": 1,           # 2024+
        "high_relevance": 1,
        "has_methodology": 1,
    },
}


# =====================================================================
# Mode: sota_survey — SOTA method comparison
# =====================================================================
SOTA_SURVEY: dict[str, Any] = {
    "name": "sota_survey",
    "label": "SOTA Survey",
    "description": "Find and compare state-of-the-art methods for a specific task.",
    "output_schema": """### Output format (MUST be a parseable JSON object)
```json
{
  "proposals": [
    {
      "name": "method_name",
      "category": "method_type (e.g. physics-based, data-driven, hybrid)",
      "source": "paper title + venue + year",
      "url": "arxiv or repo URL, null if none",
      "rationale": "why this method is notable for the task",
      "key_metric": "best reported metric and value",
      "dataset": "which benchmark dataset was used",
      "code_available": true,
      "complexity": "low|medium|high"
    }
  ]
}
```
Return ONLY the JSON object, no prose outside it.""",

    "per_ai_direction": None,  # filled dynamically
    "task_context": None,
    "extra_instructions": (
        "For each method, report the BEST published metric on a standard benchmark. "
        "Only include methods with quantitative results. Note whether code is publicly "
        "available. Compare methods fairly on the same benchmarks where possible."
    ),

    "claude_summary_prompt": (
        "You are Claude Opus acting as a senior researcher comparing SOTA methods.\n\n"
        "Create a comprehensive comparison table, rank methods by performance, "
        "analyze trade-offs (accuracy vs speed vs complexity), and recommend "
        "the top methods to try.\n\n"
        "RULES: Keep ALL original methods. Output a well-structured Markdown survey."
    ),

    "scoring": {
        "multi_scout_bonus": 3,
        "has_arxiv_or_doi": 2,
        "has_code": 2,
        "has_key_metric": 1,
        "low_complexity": 1,
    },
}


# =====================================================================
# Mode: custom — free-form research
# =====================================================================
CUSTOM: dict[str, Any] = {
    "name": "custom",
    "label": "Custom Research",
    "description": "Free-form research question with custom output schema.",
    "output_schema": None,  # must be provided by user
    "per_ai_direction": None,  # filled from research_question
    "task_context": None,
    "extra_instructions": "",
    "claude_summary_prompt": (
        "You are Claude Opus acting as a senior research advisor.\n\n"
        "Analyze the collected results, synthesize findings, identify patterns "
        "and gaps, and provide actionable recommendations.\n\n"
        "RULES: Keep ALL original entries. You may add new insights. "
        "Output a well-structured Markdown report."
    ),
    "scoring": {
        "multi_scout_bonus": 3,
    },
}


# =====================================================================
# Registry
# =====================================================================
MODES: dict[str, dict[str, Any]] = {
    "model_scout": MODEL_SCOUT,
    "literature": LITERATURE,
    "sota_survey": SOTA_SURVEY,
    "custom": CUSTOM,
}


def get_mode(name: str) -> dict[str, Any]:
    """Return mode config or raise KeyError."""
    if name not in MODES:
        raise KeyError(f"unknown mode: {name!r} (known: {list(MODES)})")
    return MODES[name]


__all__ = ["MODES", "get_mode", "MODEL_SCOUT", "LITERATURE", "SOTA_SURVEY", "CUSTOM"]
