"""Deduplicate, classify and score raw proposals from AI callers.

Each AI caller returns a list of raw proposal dicts (see ``prompt_builder.py``
for the schema). The same architecture often appears under slightly different
names (``"Attention U-Net"`` vs ``"AttentionUNet"`` vs ``"attention_unet"``),
so we normalise the name, merge duplicates, accumulate the set of scouts that
proposed each one, and score each survivor against the DESIGN.md rubric.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

LOGGER = logging.getLogger("model_scout.proposal_collector")


# -----------------------------------------------------------------------
# Name normalisation
# -----------------------------------------------------------------------
_NAME_ALIASES: dict[str, str] = {
    "u-net": "unet",
    "unet++": "unet_plus_plus",
    "attention u-net": "attention_unet",
    "attentionunet": "attention_unet",
    "res-unet": "res_unet",
    "resunet": "res_unet",
    "swin-unet": "swin_unet",
    "swinunet": "swin_unet",
    "vit": "vit",
    "vision transformer": "vit",
    "fno": "fno",
    "fourier neural operator": "fno",
    "deeponet": "deeponet",
    "mlp-mixer": "mlp_mixer",
    "mlpmixer": "mlp_mixer",
}


def normalize_name(raw: str) -> str:
    """Canonicalise an architecture name for duplicate detection.

    Lowercase, strip punctuation/whitespace, collapse separators. Then apply
    a small alias table for well-known families.
    """
    if not raw:
        return ""
    s = raw.strip().lower()
    if s in _NAME_ALIASES:
        return _NAME_ALIASES[s]
    # Replace any non-alphanumeric run with a single underscore.
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return _NAME_ALIASES.get(s, s)


# -----------------------------------------------------------------------
# Proposal object
# -----------------------------------------------------------------------
@dataclass
class Proposal:
    name: str
    category: str
    variant_of: str | None
    source: str
    url: str | None
    rationale: str
    novelty: str                # "existing" | "generated"
    estimated_params_m: float | None
    estimated_vram_gb: float | None
    difficulty: str             # "easy" | "medium" | "hard"
    scouted_by: list[str] = field(default_factory=list)
    raw_entries: list[dict[str, Any]] = field(default_factory=list)
    score: int = 0

    # ----- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "variant_of": self.variant_of,
            "source": self.source,
            "url": self.url,
            "rationale": self.rationale,
            "novelty": self.novelty,
            "estimated_params_m": self.estimated_params_m,
            "estimated_vram_gb": self.estimated_vram_gb,
            "difficulty": self.difficulty,
            "scouted_by": sorted(set(self.scouted_by)),
            "score": self.score,
        }

    def fingerprint(self) -> str:
        """Stable id across runs â€” handy for cross-campaign dedup."""
        h = hashlib.sha1(self.name.encode("utf-8")).hexdigest()[:10]
        return f"{self.name}_{h}"


# -----------------------------------------------------------------------
# Collector
# -----------------------------------------------------------------------
class ProposalCollector:
    """Accumulate raw proposals from multiple AI callers.

    Typical usage::

        collector = ProposalCollector(existing=["unet_v3", "fno_v3"])
        collector.ingest("claude", claude_result["proposals"])
        collector.ingest("deepseek", deepseek_result["proposals"])
        top = collector.rank()
    """

    def __init__(
        self,
        existing: Iterable[str] | None = None,
        architectures_in_zoo: Iterable[str] | None = None,
    ) -> None:
        self._by_name: dict[str, Proposal] = {}
        # "Existing paradigms" (categories already covered) inform the
        # diversity-bonus score. Distinct from "already in zoo".
        self._existing_names = {normalize_name(n) for n in (existing or [])}
        self._zoo_names = {normalize_name(n) for n in (architectures_in_zoo or [])}

    # --- public API ------------------------------------------------------
    def ingest(self, scout: str, raw_proposals: Iterable[dict[str, Any]]) -> int:
        """Add proposals from one AI. Returns how many were new."""
        added = 0
        for p in raw_proposals or []:
            try:
                norm = self._normalise(p)
            except Exception as exc:      # noqa: BLE001 - defensive; AI output is untrusted
                LOGGER.warning("Skipping malformed proposal from %s: %s", scout, exc)
                continue
            if norm.name in self._zoo_names:
                LOGGER.debug("Proposal %s already registered in zoo â€” skipping", norm.name)
                continue
            existing = self._by_name.get(norm.name)
            if existing is None:
                norm.scouted_by.append(scout)
                norm.raw_entries.append(p)
                self._by_name[norm.name] = norm
                added += 1
            else:
                existing.scouted_by.append(scout)
                existing.raw_entries.append(p)
                self._merge_fields(existing, norm)
        return added

    def rank(self, min_score: int = 0) -> list[Proposal]:
        """Score every proposal and return them sorted descending."""
        for prop in self._by_name.values():
            prop.score = self._score(prop)
        ranked = sorted(
            self._by_name.values(),
            key=lambda p: (p.score, -len(p.scouted_by)),
            reverse=True,
        )
        return [p for p in ranked if p.score >= min_score]

    def all(self) -> list[Proposal]:
        return list(self._by_name.values())

    # --- internals -------------------------------------------------------
    @staticmethod
    def _normalise(raw: dict[str, Any]) -> Proposal:
        name = normalize_name(raw.get("name", ""))
        if not name:
            raise ValueError("proposal missing 'name'")
        return Proposal(
            name=name,
            category=str(raw.get("category") or "other"),
            variant_of=(raw.get("variant_of") or None),
            source=str(raw.get("source") or ""),
            url=(raw.get("url") or None),
            rationale=str(raw.get("rationale") or ""),
            novelty=str(raw.get("novelty") or "existing"),
            estimated_params_m=_maybe_float(raw.get("estimated_params_m")),
            estimated_vram_gb=_maybe_float(raw.get("estimated_vram_gb")),
            difficulty=str(raw.get("difficulty") or "medium"),
        )

    @staticmethod
    def _merge_fields(existing: Proposal, incoming: Proposal) -> None:
        """When two AIs propose the same arch, keep the richest metadata."""
        if not existing.url and incoming.url:
            existing.url = incoming.url
        if not existing.source and incoming.source:
            existing.source = incoming.source
        if not existing.rationale:
            existing.rationale = incoming.rationale
        elif incoming.rationale and incoming.rationale not in existing.rationale:
            existing.rationale = f"{existing.rationale} | {incoming.rationale}"
        if existing.estimated_params_m is None:
            existing.estimated_params_m = incoming.estimated_params_m
        if existing.estimated_vram_gb is None:
            existing.estimated_vram_gb = incoming.estimated_vram_gb

    def _score(self, p: Proposal) -> int:
        """Rubric from DESIGN.md Â§'è¯„åˆ†æ ‡å‡†'."""
        score = 0
        if len(set(p.scouted_by)) >= 2:
            score += 3
        if p.url and ("arxiv" in p.url.lower() or "doi" in p.url.lower()):
            score += 2
        if p.estimated_params_m is not None and p.estimated_params_m < 50:
            score += 1
        if p.estimated_vram_gb is not None and p.estimated_vram_gb < 40:
            score += 1
        # Diversity bonus: different paradigm than anything currently in zoo.
        if p.category and p.category not in self._existing_categories():
            score += 2
        if p.novelty == "generated":
            score += 1
        return score

    def _existing_categories(self) -> set[str]:
        # UNet family â†’ cnn_encoder_decoder; FNO â†’ operator.
        defaults = {"cnn_encoder_decoder", "operator"}
        return defaults


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["Proposal", "ProposalCollector", "normalize_name"]



