"""Auto V6 Explorer â€” AI-driven experiment exploration.

Core modules:
  - explorer.py: ExperimentConfig, ExperimentResult, data loading
  - candidate_library.py: Model catalog + HP search space (NO V3 results)
  - planner.py: ExplorerPlanner (multi-AI suggestion, baseline-first workflow)
  - suggester.py: Legacy suggestion generation (warm start)
"""
from .explorer import ExperimentConfig, ExperimentResult, summarize_results
from .candidate_library import CandidateLibrary, MODEL_CATALOG, HPSpace
from .planner import ExplorerPlanner

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "load_v3_baseline",
    "summarize_results",
    "CandidateLibrary",
    "MODEL_CATALOG",
    "HPSpace",
    "ExplorerPlanner",
]

from .codegen import generate_model, generate_batch, load_specs

from .reviewer import review_round, build_review_context, RoundReview

