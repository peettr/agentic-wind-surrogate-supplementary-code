"""Sequential Planner â€” uses V3 Model Scout architecture for per-round experiment suggestion.

Flow per round:
  1. Build context (history + review + constraints + libraries)
  2. Phase 1: 7 AI scouts propose in parallel (glm/claude/codex/deepseek/mimo/gemini/grok)
  3. Phase 2: Codex CLI synthesizes all scout proposals into final 12 experiments
  4. Validate + score + return ExperimentConfigs
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from explorer.candidate_library import CandidateLibrary
from explorer.explorer import ExperimentConfig, ExperimentResult
from explorer.ai_callers import get_caller, CALLERS

LOGGER = logging.getLogger("hybrid.planner")

# ======================================================================
# Hard constraints
# ======================================================================

HARD_CONSTRAINTS = """HARD CONSTRAINTS (must not violate):
- ordinary batch_size = 16 (fixed/default; do not tune batch_size for performance)
- lower batch_size is allowed only at batch_size=8 for explicit resource_probe/OOM repair/resource_guard safe-config paths, and those runs are feasibility evidence rather than ordinary leaderboard candidates unless rerun/normalized; batch_size<8 requires manual_resource_probe_approved=True and must not be auto-suggested
- data_augment = false (V3 proved this breaks physics alignment)
- Input: (B, C, 640, 640), Output: (B, 1, 640, 640) with ReLU (non-negative)
- GroupNorm (not BatchNorm, EMA compatible)
- n_c options: 16, 32 (V3 validated range)
- max params <= 150M
- seed = 1 (exploration phase)
- loss must be registered in V3's LossLibrary: masked_l1, masked_l1_gradient, masked_huber
"""

EXPERIMENT_SCHEMA = """Return ONLY a JSON array of exactly 12 experiment suggestions:
[
  {
    "arch_name": "model_name",
    "n_c": 16,
    "lr": 0.001,
    "loss_name": "masked_l1",
    "input_features": "height",
    "use_ema": false,
    "ema_decay": 0.999,
    "scheduler": null,
    "weight_decay": 0,
    "gradient_clip": null,
    "depth": 7,
    "reason": "1-2 sentences: why this experiment is likely to improve R2"
  }
]"""


# ======================================================================
# Prompt builders
# ======================================================================

def build_model_library_text(library: CandidateLibrary, history: list) -> str:
    tried = {}
    for r in history:
        arch = getattr(r, "arch_name", "") or (r.config.arch_name if hasattr(r, "config") else "")
        r2 = (getattr(r, "val_r2_median", None) if hasattr(r, "val_r2_median") and r.val_r2_median is not None else None)
        if arch and r2 is not None:
            if arch not in tried or r2 > tried[arch]:
                tried[arch] = r2

    lines = ["CANDIDATE MODELS (strongly recommended, but you may propose new ones):", ""]
    for name, spec in library.models.items():
        if not spec.enabled:
            continue
        best = tried.get(name)
        best_str = f" | best R2={best:.4f}" if best is not None else " | not tested"
        lines.append(f"  {name} ({spec.category}, ~{spec.params_million}M, in_ch={spec.input_channels}){best_str}")

    lines.append("")
    lines.append("You may propose models not listed above if you believe a different")
    lines.append("architecture could improve R2. Explain your reasoning.")
    return "\n".join(lines)


def build_hp_options(library: CandidateLibrary) -> str:
    hp = library.hp_space
    return f"""RECOMMENDED HP OPTIONS (you may suggest beyond these):
- loss: {', '.join(hp.loss_name)}
- lr: {', '.join(str(x) for x in hp.lr)} (baseline from V3)
- n_c: {', '.join(str(x) for x in hp.n_c)}
- input_features: {', '.join(hp.input_features)}
- EMA: null or 0.999
- scheduler: null or cosine
- weight_decay: 0 or 1e-4
- gradient_clip: null or 0.5

You may suggest HPs not listed above if you believe they could improve R2."""


def build_history_text(history: list, max_rows: int = 20) -> str:
    def _arch(r):
        return r.config.arch_name if hasattr(r, "config") else ""
    def _r2(r):
        v = getattr(r, "r2_median", None)
        return v if v is not None and v == v else None
    def _status(r):
        return getattr(r, "status", "")
    def _loss(r):
        return r.config.loss_name if hasattr(r, "config") else ""
    def _lr(r):
        return r.config.lr if hasattr(r, "config") else 0
    def _input(r):
        return r.config.input_features if hasattr(r, "config") else ""
    def _epochs(r):
        return r.config.epochs if hasattr(r, "config") else 0
    def _nc(r):
        return r.config.n_c if hasattr(r, "config") else 0

    completed = [r for r in history if _status(r) in ("ok", "completed") and _r2(r) is not None]
    completed.sort(key=lambda r: _r2(r), reverse=True)

    baseline = [r for r in completed if _arch(r) == "unet_v2_baseline"]
    if len(completed) <= 1 and baseline:
        b = baseline[0]
        return (
            "BASELINE (starting point - must beat this):\n"
            "  Model: unet_v2_baseline (7-level UNet)\n"
            "  Config: n_c=16, depth=7, lr=5e-4, loss=masked_l1, input=height, no EMA\n"
            f"  Result: R2={_r2(b):.4f}, epochs={_epochs(b)}\n\n"
            "V3 REFERENCE (what we are aiming to beat):\n"
            "  V3 best height-only: DilatedUNet R2=0.708 (lr=5e-4, masked_l1)\n"
            "  V3 best with SDF: UNet 3ch R2=0.724 (height+SDF+normal)\n"
            "  Target: R2 > 0.724\n\n"
            "No other experiments yet. Propose the first batch."
        )

    if not completed:
        return "No completed experiments yet."

    lines = [f"EXPERIMENT HISTORY ({len(completed)} completed, top {min(len(completed), max_rows)}):", ""]
    lines.append(f"{'Model':<25s} {'n_c':>4s} {'lr':>8s} {'Loss':<22s} {'Input':<18s} {'R2':>8s} {'Ep':>4s}")
    lines.append("-" * 90)
    for r in completed[:max_rows]:
        lines.append(f"{_arch(r):<25s} {_nc(r):>4d} {_lr(r):>8.0e} {_loss(r):<22s} {_input(r):<18s} {_r2(r):>8.4f} {_epochs(r):>4d}")
    if len(completed) > max_rows:
        lines.append(f"  ... and {len(completed) - max_rows} more")
    return chr(10).join(lines)

def build_review_text(review: Optional[dict]) -> str:
    if not review:
        return ""
    lines = ["", "PREVIOUS ROUND REVIEW:", ""]
    lines.append(f"Best: {review.get('best_arch', '?')} R2={review.get('best_r2', 0):.4f}")
    change = review.get("r2_vs_previous", 0)
    if change:
        direction = "improved" if change > 0 else "regressed"
        lines.append(f"Change: {change:+.4f} ({direction})")
    if review.get("issues"):
        lines.append("Issues:")
        for i in review["issues"]:
            lines.append(f"  - {i}")
    if review.get("recommendations"):
        lines.append("Recommendations:")
        for r in review["recommendations"]:
            lines.append(f"  - {r}")
    if review.get("stagnation_detected"):
        lines.append("WARNING: Stagnation detected. Consider a fundamentally different direction.")
    return "\n".join(lines)


# ======================================================================
# JSON extraction
# ======================================================================

def extract_json_array(text: str) -> list[dict]:
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return []


def validate_proposal(prop: dict, library: CandidateLibrary) -> list[str]:
    issues = []
    arch = prop.get("arch_name", "")
    if not arch:
        issues.append("missing arch_name")
    n_c = prop.get("n_c", 16)
    if n_c not in (8, 16, 24, 32, 48):
        issues.append(f"unusual n_c={n_c}")
    lr = prop.get("lr", 1e-3)
    if lr <= 0 or lr > 0.1:
        issues.append(f"invalid lr={lr}")
    if prop.get("loss_name", "masked_l1") not in {"masked_l1", "masked_l1_gradient", "masked_huber"}:
        issues.append(f"unknown loss={prop.get('loss_name')}")
    if prop.get("augmentation", False):
        issues.append("augmentation must be false")
    if prop.get("input_features", "height") not in {"height", "height_sdf", "height_sdf_normal"}:
        issues.append(f"unknown input={prop.get('input_features')}")
    return issues


# ======================================================================
# Main Planner
# ======================================================================

class V4Planner:
    """Sequential Planner using V3 Model Scout architecture.
    
    Per round:
      Phase 1: 7 AI scouts propose in parallel
      Phase 2: Codex CLI synthesizes final 12 experiments
    """

    def __init__(self, library: CandidateLibrary, campaign_dir: Path,
                 ai_models: list[str] | None = None, n_proposals: int = 12):
        self.library = library
        self.campaign_dir = Path(campaign_dir)
        # Default: all V3 scouts except deep_research
        self.ai_models = ai_models or [m for m in CALLERS.keys() if m != "deep_research"]
        self.n_proposals = n_proposals
        self.round_num = 0

    def propose_experiments(self, history: list[ExperimentResult],
                            review: Optional[dict] = None) -> list[ExperimentConfig]:
        self.round_num += 1
        LOGGER.info("=== V4Planner Round %d ===", self.round_num)

        context = self._build_context(history, review)
        raw_responses = self._call_ai(context)

        all_proposals = []
        for ai_name, response in raw_responses:
            props = extract_json_array(response)
            LOGGER.info("  %s: %d proposals", ai_name, len(props))
            all_proposals.extend(props)

        if not all_proposals:
            LOGGER.warning("No proposals from any AI")
            return []

        valid = [p for p in all_proposals if not validate_proposal(p, self.library)]
        LOGGER.info("  %d/%d passed validation", len(valid), len(all_proposals))

        scored = self._score_proposals(valid, history)

        configs = []
        seen = set()
        for p in scored[:self.n_proposals]:
            arch = p.get("arch_name", "")
            if arch in seen:
                continue
            seen.add(arch)
            spec = self.library.models.get(arch)
            cfg = ExperimentConfig(
                arch_name=arch,
                n_c=p.get("n_c", spec.n_c_options[-1] if spec and spec.n_c_options else 16),
                depth=p.get("depth", spec.depth_options[-1] if spec and spec.depth_options else 7),
                loss_name=p.get("loss_name", "masked_l1"),
                lr=p.get("lr", 1e-3),
                scheduler=p.get("scheduler"),
                weight_decay=p.get("weight_decay", 0),
                gradient_clip=p.get("gradient_clip"),
                use_ema=p.get("use_ema", False),
                ema_decay=p.get("ema_decay", 0.999),
                augmentation=False,
                input_features=p.get("input_features", "height"),
                epochs=p.get("epochs", 200),
                seed=1,
            )
            configs.append(cfg)
            LOGGER.info("  -> %s n_c=%d lr=%.0e loss=%s input=%s",
                        cfg.arch_name, cfg.n_c, cfg.lr, cfg.loss_name, cfg.input_features)

        # Save proposals
        proposal_dir = self.campaign_dir / "proposals"
        proposal_dir.mkdir(parents=True, exist_ok=True)
        (proposal_dir / f"round_{self.round_num:04d}.json").write_text(
            json.dumps(scored[:self.n_proposals], indent=2, ensure_ascii=False), encoding="utf-8")

        return configs

    # ------------------------------------------------------------------

    def _build_context(self, history, review, epochs: int = 200) -> str:
        parts = [
            "You are a machine learning researcher optimizing a surrogate model",
            "for urban wind pressure prediction (height map to wind speed field).",
            "",
            HARD_CONSTRAINTS,
            "",
            build_model_library_text(self.library, history),
            "",
            build_hp_options(self.library),
            "",
            build_history_text(history),
            build_review_text(review),
            "",
            f"Propose {self.n_proposals} experiments for the next round. All should use {epochs} epochs.",
            "Diversify: try different models, HPs, and input features.",
            "Build on what worked well in previous rounds.",
            "",
            EXPERIMENT_SCHEMA,
        ]
        return "\n".join(parts)

    def _call_ai(self, context: str) -> list[tuple[str, str]]:
        """Phase 1: 7 scouts in parallel. Phase 2: Codex synthesizes final 12."""
        import concurrent.futures as futures

        scout_results = []

        def _call_one(ai_name: str):
            caller = get_caller(ai_name)
            try:
                response = caller(context, timeout=600)
                proposals = response.get("proposals", [])
                return ai_name, json.dumps(proposals, ensure_ascii=False) if proposals else ""
            except Exception as e:
                LOGGER.warning("Scout %s failed: %s", ai_name, e)
                return ai_name, ""

        with futures.ThreadPoolExecutor(max_workers=min(8, len(self.ai_models))) as pool:
            futs = {pool.submit(_call_one, m): m for m in self.ai_models}
            for f in futures.as_completed(futs):
                ai_name, text = f.result()
                if text:
                    scout_results.append((ai_name, text))
                    LOGGER.info("  scout %s: received", ai_name)
                else:
                    LOGGER.warning("  scout %s: empty", ai_name)

        if not scout_results:
            LOGGER.warning("All scouts failed")
            return []

        # Phase 2: Codex synthesizes
        LOGGER.info("  Synthesizing via Codex CLI...")
        scout_block = ""
        for name, proposals_json in scout_results:
            scout_block += "\n### " + name + "\n" + proposals_json + "\n"

        n = self.n_proposals
        synthesis_prompt = (
            "You are a senior ML researcher. Synthesize experiment proposals "
            "from multiple AI scouts into the best " + str(n) + " experiments.\n\n"
            "CONTEXT:\n" + context + "\n\n"
            "SCOUT PROPOSALS:\n" + scout_block + "\n"
            "Select and refine " + str(n) + " experiments. Remove duplicates, "
            "check constraints, prioritize diversity. You may adjust HPs slightly.\n\n"
            "Return ONLY a JSON array of exactly " + str(n) + " proposals."
        )

        try:
            codex_caller = get_caller("codex")
            response = codex_caller(synthesis_prompt, timeout=900)
            final = response.get("proposals", [])
            if final:
                LOGGER.info("  Codex synthesis: %d final proposals", len(final))
                return [("codex_synth", json.dumps(final, ensure_ascii=False))]
        except Exception as e:
            LOGGER.warning("  Codex synthesis failed: %s", e)

        # Fallback
        all_p = []
        for _, pj in scout_results:
            try:
                all_p.extend(json.loads(pj))
            except Exception:
                pass
        return [("merged", json.dumps(all_p[:self.n_proposals], ensure_ascii=False))]

    def _score_proposals(self, proposals: list[dict], history: list) -> list[dict]:
        tried_models = set()
        best_r2 = {}
        tried_configs = set()
        for r in history:
            if getattr(r, "status", "") == "ok":
                arch = getattr(r, "arch_name", "") or (r.config.arch_name if hasattr(r, "config") else "")
                tried_models.add(arch)
                r2 = (getattr(r, "val_r2_median", None) if hasattr(r, "val_r2_median") and r.val_r2_median is not None else None)
                if r2 is not None and (arch not in best_r2 or r2 > best_r2[arch]):
                    best_r2[arch] = r2
                tried_configs.add((arch, getattr(r, "loss_name", "") or (r.config.loss_name if hasattr(r, "config") else ""), getattr(r, "input_features", "") or (r.config.input_features if hasattr(r, "config") else "")))

        for p in proposals:
            score = 0
            arch = p.get("arch_name", "")
            if arch not in tried_models:
                score += 3
            inp = p.get("input_features", "height")
            if inp != "height" and arch in best_r2 and best_r2[arch] > 0.65:
                score += 2
            if arch in best_r2:
                score += min(int(best_r2[arch] * 5), 5)
            config_key = (arch, p.get("loss_name", "masked_l1"), p.get("input_features", "height"))
            if config_key not in tried_configs:
                score += 1
            if p.get("use_ema") and arch in best_r2 and best_r2[arch] > 0.65:
                score += 1
            if len(p.get("reason", "")) > 50:
                score += 1
            p["_score"] = score

        proposals.sort(key=lambda p: p.get("_score", 0), reverse=True)
        return proposals


__all__ = ["V4Planner"]




