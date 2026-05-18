"""StateManager: atomic campaign-state persistence.

Responsibilities:

* Atomic writes via temp file + ``os.replace`` (no torn reads).
* Load-or-create of :class:`CampaignState`.
* Append-only history log (``history.jsonl``).
* Manifest pinning ``config_hash`` / ``eval_hash`` / ``split_hash``.
* Kill switch: the presence of ``HALT`` in the campaign dir aborts the run.
* Failure budget helper (configurable threshold, default 50 %).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from shared.configs.schema import (
    BaselineMetrics,
    CampaignState,
    ChampionSnapshot,
    ExperimentResult,
)


LOGGER = logging.getLogger("auto_v3.state")


class KillSwitchError(RuntimeError):
    """Raised when the ``HALT`` sentinel is found in the campaign dir."""


class StateManager:
    """Durable, atomic state store for a single campaign."""

    def __init__(
        self,
        campaign_dir: str | Path,
        baseline: BaselineMetrics,
        strategy: str,
        campaign_id: Optional[str] = None,
    ) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.campaign_dir / "state.json"
        self.manifest_path = self.campaign_dir / "manifest.json"
        self.history_path = self.campaign_dir / "history.jsonl"
        self.halt_path = self.campaign_dir / "HALT"
        self.baseline = baseline
        self.strategy = strategy
        self.campaign_id = campaign_id or (
            f"{strategy}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------
    def check_kill_switch(self) -> None:
        if self.halt_path.exists():
            raise KillSwitchError(f"HALT file present: {self.halt_path}")

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------
    def load_or_create(self) -> CampaignState:
        if self.state_path.exists():
            state = CampaignState.model_validate_json(
                self.state_path.read_text()
            )
            LOGGER.info("Loaded campaign state from %s", self.state_path)
            return state
        state = CampaignState(
            campaign_id=self.campaign_id,
            strategy=self.strategy,
            baseline=self.baseline,
        )
        self.save(state)
        return state

    def load(self) -> CampaignState:
        return CampaignState.model_validate_json(self.state_path.read_text())

    def save(self, state: CampaignState) -> None:
        self.check_kill_switch()
        state.updated_at = datetime.utcnow().isoformat() + "Z"
        self._atomic_write(self.state_path, state.model_dump_json(indent=2))

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False, suffix=".tmp",
        )
        try:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, path)
        except Exception:
            tmp.close()
            try:
                os.unlink(tmp.name)
            except FileNotFoundError:
                pass
            raise

    # ------------------------------------------------------------------
    # History (history.jsonl is the SINGLE SOURCE OF TRUTH for results)
    # ------------------------------------------------------------------
    def append_history(self, result: ExperimentResult) -> None:
        """Append a result line to history.jsonl with fsync + flush.

        This is not fully atomic (rename-in-place would require rewriting
        the whole file), but fsync guarantees that a crashed process does
        not leave a half-flushed line to confuse the next reader.
        """
        line = result.model_dump_json() + "\n"
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.history_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644,
        )
        try:
            os.write(fd, line.encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def get_history(self) -> list[ExperimentResult]:
        if not self.history_path.exists():
            return []
        out: list[ExperimentResult] = []
        with open(self.history_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(ExperimentResult.model_validate_json(line))
                except Exception as exc:  # tolerate last-line torn writes
                    LOGGER.warning("Skipping malformed history line: %s", exc)
        return out

    # ------------------------------------------------------------------
    # Champion / failures
    # ------------------------------------------------------------------
    def set_champion(
        self,
        state: CampaignState,
        experiment_id: str,
        snapshot: Optional[ChampionSnapshot] = None,
    ) -> None:
        """Record the champion id (and optionally a full snapshot)."""
        state.champion_id = experiment_id
        if snapshot is not None:
            state.champion = snapshot
        self.save(state)

    def record_failure(self, state: CampaignState) -> None:
        state.failure_count += 1
        self.save(state)

    def failure_rate_exceeded(
        self, state: CampaignState, threshold: float = 0.5,
    ) -> bool:
        if state.total_submitted == 0:
            return False
        return state.failure_count / state.total_submitted > threshold

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def write_manifest(
        self,
        state: CampaignState,
        config_hash: str,
        eval_hash: str,
        split_hash: str,
    ) -> None:
        state.config_hash = config_hash
        state.eval_hash = eval_hash
        state.split_hash = split_hash
        manifest = {
            "campaign_id": state.campaign_id,
            "strategy": state.strategy,
            "config_hash": config_hash,
            "eval_hash": eval_hash,
            "split_hash": split_hash,
            "baseline": state.baseline.model_dump(),
            "started_at": state.started_at,
        }
        self._atomic_write(
            self.manifest_path, json.dumps(manifest, indent=2),
        )
        self.save(state)


__all__ = ["KillSwitchError", "StateManager"]
