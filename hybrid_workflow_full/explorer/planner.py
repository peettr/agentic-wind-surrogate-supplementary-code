"""Explorer Planner for Hybrid - AI-driven experiment suggestion.

Key principles (the human researcher's corrections):
  1. Smoke tests ONLY verify code correctness, NOT for ranking
  2. No V3 results used - Sequential discovers everything from scratch
  3. Start with baseline (unet_v2_baseline) hyperparameter tuning
  4. Each round calls multi-AI to generate suggestions
  5. No predetermined model assignments per round
  6. Candidate library is reference only, Explorer can go beyond it

Multi-AI suggestion system:
  - Primary: Claude CLI (claude-opus), Codex CLI (gpt-5.4)
  - Secondary: Gemini, Grok, GLM, MiMo, DeepSeek
  - Each round, the Planner generates an AI prompt with current state,
    gets suggestions, and incorporates them into the next batch

Workflow:
  Phase 1 BASELINE: Tune unet_v2_baseline HPs (3-4 rounds)
  Phase 2 EXPLORE: Broad model exploration (smoke 20ep)
  Phase 3 FOCUS: 200ep on promising models
  Phase 4 LONG: 1000ep on top candidates
  Phase 5 MULTI_SEED: seed=2,3 for top 3
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from explorer.candidate_library import CandidateLibrary, ModelSpec, HPSpace
from explorer.explorer import ExperimentConfig, ExperimentResult


LOGGER = logging.getLogger("hybrid.planner")

MAX_ROUNDS = 12
MAX_PER_ROUND = 12
MAX_RUNS_PER_MODEL = 5
SMOKE_EPOCHS = 20
BASELINE_EPOCHS = 200
FULL_EPOCHS = 200
LONG_EPOCHS = 1000

# AI model configuration for suggestion generation
AI_PRIMARY = ["claude", "codex"]  # Claude Opus + GPT-5.4
AI_SECONDARY = ["gemini", "grok", "glm", "mimo", "deepseek"]


class ExplorerPlanner:
    """AI-driven planner with multi-AI suggestion per round."""

    def __init__(
        self,
        library: CandidateLibrary,
        campaign_dir: Path,
        max_rounds: int = MAX_ROUNDS,
        max_per_round: int = MAX_PER_ROUND,
    ):
        self.library = library
        self.campaign_dir = Path(campaign_dir)
        self.max_rounds = max_rounds
        self.max_per_round = max_per_round

        # State
        self.round_num = 0
        self.phase = "baseline"  # baseline, smoke, focus, long, multi_seed
        self.model_run_count: dict[str, int] = {}
        self.baseline_done = False
        self._stagnation = 0
        self._last_best = float("-inf")
        self._ai_suggestions: list[dict] = []  # cached AI suggestions per round

    # ------------------------------------------------------------------
    # BasePlanner interface
    # ------------------------------------------------------------------

    def propose_experiments(
        self,
        history: list[ExperimentResult],
        baseline=None,
    ) -> list[ExperimentConfig]:
        self.round_num += 1
        LOGGER.info("Explorer round %d (phase=%s)", self.round_num, self.phase)
        self._update_counts(history)
        self._check_phase_transition(history)

        if self.phase == "baseline":
            return self._propose_baseline(history)
        elif self.phase == "smoke":
            return self._propose_smoke(history)
        elif self.phase == "focus":
            return self._propose_focus(history)
        elif self.phase == "long":
            return self._propose_long(history)
        else:
            return []

    def is_done(self, history=None, baseline=None) -> bool:
        if self.round_num >= self.max_rounds:
            return True
        if self.phase == "long" and self.round_num > 1:
            return True
        return False

    def get_state(self) -> dict:
        return {
            "round_num": self.round_num,
            "phase": self.phase,
            "model_run_count": self.model_run_count,
            "baseline_done": self.baseline_done,
            "stagnation": self._stagnation,
            "last_best": self._last_best,
            "ai_suggestions": self._ai_suggestions,
        }

    def restore_state(self, state: dict) -> None:
        self.round_num = state.get("round_num", 0)
        self.phase = state.get("phase", "baseline")
        self.model_run_count = state.get("model_run_count", {})
        self.baseline_done = state.get("baseline_done", False)
        self._stagnation = state.get("stagnation", 0)
        self._last_best = state.get("last_best", float("-inf"))
        self._ai_suggestions = state.get("ai_suggestions", [])

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def _check_phase_transition(self, history: list[ExperimentResult]):
        completed = [r for r in history if self._is_ok(r)]

        # baseline -> smoke: after baseline completes (1+ successful baseline runs)
        if self.phase == "baseline" and self.baseline_done:
            self.phase = "smoke"
            LOGGER.info("Baseline done. Transitioning to SMOKE.")

        # smoke -> focus: after enough smoke tests complete (>= 12 or round >= 4)
        if self.phase == "smoke":
            smoke_done = [r for r in completed if self._get_epochs(r) <= SMOKE_EPOCHS]
            if len(smoke_done) >= 12 or self.round_num >= 5:
                self.phase = "focus"
                LOGGER.info("Smoke complete (%d runs). Transitioning to FOCUS.", len(smoke_done))

        # focus -> long: after enough focus runs (>= 12 or round >= 8)
        if self.phase == "focus":
            focus_done = [r for r in completed if self._get_epochs(r) == FULL_EPOCHS]
            if len(focus_done) >= 12 or self.round_num >= 9:
                self.phase = "long"
                LOGGER.info("Focus complete (%d runs). Transitioning to LONG.", len(focus_done))

    def _update_counts(self, history: list[ExperimentResult]):
        self.model_run_count.clear()
        for r in history:
            arch = getattr(r, "arch_name", "unknown")
            self.model_run_count[arch] = self.model_run_count.get(arch, 0) + 1

    def _is_ok(self, r: ExperimentResult) -> bool:
        return getattr(r, "status", "") == "ok" and getattr(r, "val_r2_median", None) is not None

    def _get_epochs(self, r: ExperimentResult) -> int:
        return getattr(r, "epochs", 0) or 0

    def _can_run(self, model_name: str) -> bool:
        return self.model_run_count.get(model_name, 0) < MAX_RUNS_PER_MODEL

    # ------------------------------------------------------------------
    # Phase: BASELINE - tune unet_v2_baseline hyperparameters
    # ------------------------------------------------------------------

    def _propose_baseline(self, history: list[ExperimentResult]) -> list[ExperimentConfig]:
        """Start with baseline UNet HP tuning. This is the starting point."""
        experiments = []

        # V3 baseline config (the anchor)
        experiments.append(ExperimentConfig(
            arch_name="unet_v2_baseline",
            n_c=16, depth=7,
            loss_name="masked_l1", lr=1e-3,
            input_features="height",
            epochs=BASELINE_EPOCHS, seed=1,
        ))

        # Try different HPs on baseline
        baseline_configs = [
            # V3 orthogonal exploratory sweep top performers (as starting points for Explorer to beat)
            {"loss_name": "masked_l1_gradient", "lr": 1e-3, "scheduler": "cosine"},
            {"loss_name": "masked_l1", "lr": 1e-3, "use_ema": True, "ema_decay": 0.999},
            {"loss_name": "masked_l1_gradient", "lr": 1e-3, "use_ema": True, "ema_decay": 0.999},
            # SDF input on baseline
            {"loss_name": "masked_l1", "lr": 1e-3, "input_features": "height_sdf_normal"},
            {"loss_name": "masked_l1_gradient", "lr": 1e-3, "input_features": "height_sdf_normal"},
            # Wider baseline
            {"n_c": 32, "loss_name": "masked_l1", "lr": 1e-3, "use_ema": True},
            {"n_c": 32, "loss_name": "masked_l1", "lr": 1e-3, "input_features": "height_sdf_normal"},
        ]

        for cfg in baseline_configs:
            if len(experiments) >= self.max_per_round:
                break
            experiments.append(ExperimentConfig(
                arch_name="unet_v2_baseline",
                n_c=cfg.get("n_c", 16),
                depth=cfg.get("depth", 7),
                loss_name=cfg.get("loss_name", "masked_l1"),
                lr=cfg.get("lr", 1e-3),
                scheduler=cfg.get("scheduler"),
                use_ema=cfg.get("use_ema", False),
                ema_decay=cfg.get("ema_decay", 0.999),
                augmentation=cfg.get("augmentation", False),
                input_features=cfg.get("input_features", "height"),
                epochs=BASELINE_EPOCHS, seed=1,
            ))

        # Mark baseline as done after this round
        self.baseline_done = True
        LOGGER.info("Baseline round: %d configs proposed", len(experiments))
        return experiments

    # ------------------------------------------------------------------
    # Phase: SMOKE - code correctness check only (20ep)
    # ------------------------------------------------------------------

    def _propose_smoke(self, history: list[ExperimentResult]) -> list[ExperimentConfig]:
        """Smoke test: verify models run without errors. NOT for ranking."""
        experiments = []

        # Get models that haven't been smoke-tested yet
        smoke_tested = set()
        for r in history:
            if self._get_epochs(r) <= SMOKE_EPOCHS:
                smoke_tested.add(getattr(r, "arch_name", ""))

        # AI suggestions are PRIMARY driver
        ai_ideas = self._get_ai_suggestions(history)

        models_to_try = []

        # 1. AI-suggested models first (highest priority)
        for idea in ai_ideas:
            arch = idea.get("arch_name", "")
            if arch in self.library.models and self._can_run(arch) and arch not in smoke_tested:
                models_to_try.append(self.library.models[arch])
                smoke_tested.add(arch)

        # 2. Fallback: remaining untested models from library
        if len(models_to_try) < 4:
            for m in self.library.get_enabled_models():
                if m.name not in smoke_tested and self._can_run(m.name) and m not in models_to_try:
                    models_to_try.append(m)
                    if len(models_to_try) >= self.max_per_round:
                        break

        # Generate smoke configs
        for m in models_to_try:
            if len(experiments) >= self.max_per_round:
                break
            # Smoke: use default n_c, height-only input (to isolate code correctness)
            experiments.append(ExperimentConfig(
                arch_name=m.name,
                n_c=m.n_c_options[0] if m.n_c_options else 16,
                depth=m.depth_options[0] if m.depth_options else 7,
                loss_name="masked_l1",
                lr=1e-3,
                input_features="height",  # height-only for smoke (no SDF confound)
                epochs=SMOKE_EPOCHS,
                seed=1,
            ))

        LOGGER.info("Smoke round %d: %d models to test", self.round_num, len(experiments))
        return experiments

    # ------------------------------------------------------------------
    # Phase: FOCUS - 200ep on validated models
    # ------------------------------------------------------------------

    def _propose_focus(self, history: list[ExperimentResult]) -> list[ExperimentConfig]:
        """Focus phase: 200ep on models that passed smoke (code works)."""
        experiments = []
        ai_ideas = self._get_ai_suggestions(history)

        # Models that passed smoke (completed 20ep without error)
        smoke_passed = set()
        for r in history:
            if self._is_ok(r) and self._get_epochs(r) <= SMOKE_EPOCHS:
                smoke_passed.add(getattr(r, "arch_name", ""))

        # Try each passed model with different HPs
        for arch in smoke_passed:
            if not self._can_run(arch):
                continue
            m = self.library.models.get(arch)
            if not m:
                continue

            # Try SDF input
            experiments.append(ExperimentConfig(
                arch_name=arch,
                n_c=m.n_c_options[-1] if m.n_c_options else 32,  # widest default
                depth=m.depth_options[-1] if m.depth_options else 7,
                loss_name="masked_l1",
                lr=1e-3,
                input_features="height_sdf_normal",
                epochs=FULL_EPOCHS, seed=1,
            ))

            if len(experiments) >= self.max_per_round:
                break

            # Try with gradient loss
            experiments.append(ExperimentConfig(
                arch_name=arch,
                n_c=m.n_c_options[-1] if m.n_c_options else 32,
                depth=m.depth_options[-1] if m.depth_options else 7,
                loss_name="masked_l1_gradient",
                lr=1e-3,
                input_features="height_sdf_normal",
                epochs=FULL_EPOCHS, seed=1,
            ))

            if len(experiments) >= self.max_per_round:
                break

        # AI suggestions are PRIMARY driver for focus phase
        for idea in ai_ideas:
            if len(experiments) >= self.max_per_round:
                break
            arch = idea.get("arch_name", "")
            m = self.library.models.get(arch)
            if m and self._can_run(arch):
                experiments.append(ExperimentConfig(
                    arch_name=arch,
                    n_c=idea.get("n_c", m.n_c_options[-1] if m.n_c_options else 32),
                    depth=idea.get("depth", m.depth_options[-1] if m.depth_options else 7),
                    loss_name=idea.get("loss_name", "masked_l1"),
                    lr=idea.get("lr", 1e-3),
                    input_features=idea.get("input_features", "height_sdf_normal"),
                    use_ema=idea.get("use_ema", False),
                    ema_decay=idea.get("ema_decay", 0.999),
                    epochs=FULL_EPOCHS, seed=1,
                ))

        # Fallback: if AI suggestions are few, use hardcoded heuristics
        if len(experiments) < 4:
            for arch in smoke_passed:
                if not self._can_run(arch):
                    continue
                m = self.library.models.get(arch)
                if not m:
                    continue
                if any(e.arch_name == arch and e.input_features == "height_sdf_normal" for e in experiments):
                    continue
                experiments.append(ExperimentConfig(
                    arch_name=arch,
                    n_c=m.n_c_options[-1] if m.n_c_options else 32,
                    depth=m.depth_options[-1] if m.depth_options else 7,
                    loss_name="masked_l1",
                    lr=1e-3,
                    input_features="height_sdf_normal",
                    epochs=FULL_EPOCHS, seed=1,
                ))
                if len(experiments) >= self.max_per_round:
                    break

        LOGGER.info("Focus round %d: %d experiments", self.round_num, len(experiments))
        return experiments

    # ------------------------------------------------------------------
    # Phase: LONG - 1000ep on top candidates
    # ------------------------------------------------------------------

    def _propose_long(self, history: list[ExperimentResult]) -> list[ExperimentConfig]:
        """1000ep on the best models found so far."""
        # Sort all 200ep results by R2
        full_results = [r for r in history if self._is_ok(r) and self._get_epochs(r) == FULL_EPOCHS]
        full_results.sort(key=lambda r: r.val_r2_median, reverse=True)

        seen = set()
        experiments = []
        for r in full_results:
            arch = getattr(r, "arch_name", "unknown")
            if arch in seen or not self._can_run(arch):
                continue
            seen.add(arch)
            experiments.append(ExperimentConfig(
                arch_name=arch,
                n_c=getattr(r, "n_c", 32),
                depth=getattr(r, "depth", 7),
                loss_name=getattr(r, "loss_name", "masked_l1"),
                lr=getattr(r, "lr", 1e-3),
                use_ema=getattr(r, "use_ema", False),
                ema_decay=getattr(r, "ema_decay", 0.999),
                input_features=getattr(r, "input_features", "height_sdf_normal"),
                epochs=LONG_EPOCHS, seed=1,
            ))
            if len(experiments) >= 10:
                break

        LOGGER.info("Long round: %d experiments (1000ep each)", len(experiments))
        return experiments

    # ------------------------------------------------------------------
    # Multi-AI Suggestion System
    # ------------------------------------------------------------------

    def _get_ai_suggestions(self, history: list[ExperimentResult]) -> list[dict]:
        """Generate experiment suggestions using multi-AI system.

        Each round, the Planner:
        1. Summarizes current state (completed runs, best results)
        2. Sends to primary AI (Claude/Codex) with candidate library
        3. Gets back experiment suggestions
        4. Validates suggestions against library (model exists, HP in range)
        5. Caches suggestions for this round

        If AI is unavailable, falls back to random exploration from library.
        """
        # For now, return cached or empty (AI integration in next step)
        # This will be wired to actual AI calls via scripts/generate_hybrid_round.py
        return self._ai_suggestions

    def set_ai_suggestions(self, suggestions: list[dict]):
        """Set AI suggestions for the next propose_experiments call.
        Called by the external AI integration layer."""
        self._ai_suggestions = suggestions

    def generate_ai_prompt(self, history: list[ExperimentResult]) -> str:
        """Generate the prompt to send to AI models for suggestions.

        The prompt includes:
        - Current round and phase
        - All completed experiment results (arch, HPs, R2)
        - Candidate library summary (model names, categories)
        - Constraints (max 12 per round, max 5 per model, etc.)
        - Request for next-round suggestions
        """
        completed = [r for r in history if self._is_ok(r)]
        completed.sort(key=lambda r: r.val_r2_median, reverse=True)

        prompt = f"""Hybrid Explorer - Round {self.round_num} ({self.phase} phase)

Current state:
- Completed {len(completed)} experiments
- Phase: {self.phase}
- Models tried: {len(set(r.arch_name for r in completed))}

Top results so far:
"""
        for i, r in enumerate(completed[:10]):
            prompt += f"  {i+1}. {r.arch_name} | loss={r.loss_name} lr={r.lr} input={r.input_features} -> R2={r.val_r2_median:.4f}\n"

        prompt += f"""
Available models (from library): {', '.join(m.name for m in self.library.get_enabled_models())}

HP space:
- loss: {self.library.hp_space.loss_name}
- lr: {self.library.hp_space.lr}
- input: {self.library.hp_space.input_features}
- n_c options: {self.library.hp_space.n_c}
- depth: per-model (from library specs)

Constraints:
- Max {self.max_per_round} experiments per round
- Max {MAX_RUNS_PER_MODEL} runs per model
- Smoke (20ep) = code correctness check only, NOT for ranking
- Focus (200ep) = actual performance evaluation
- Seed=1 for all exploration runs

Please suggest {self.max_per_round} experiments for the next round. Consider:
1. Which models haven't been smoke-tested yet?
2. Which smoke-passed models should get 200ep runs?
3. What HP combinations haven't been tried?
4. What input features (height / height_sdf / height_sdf_normal) should be tried?
5. Any creative combinations the library doesn't suggest?

Format each suggestion as:
  arch_name | n_c | depth | loss | lr | input_features | epochs | reason
"""
        return prompt

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------


__all__ = ["ExplorerPlanner"]




