"""Analyzer: aggregate per-seed ``metrics.json`` into per-experiment reports.

Reads only ``metrics.json`` (never stdout / log files) and writes:

* ``results.tsv``   â€” compact index, one row per seeded run.
* ``results.jsonl`` â€” append-only event log of :class:`ExperimentResult`.
* ``analysis.json`` â€” per-experiment aggregation (mean / std / min / max
  of val-RÂ² median across seeds, plus % improvement over baseline).

Hash cross-check helpers ensure every seed was evaluated with the same
``eval_module.py`` and ``split_manifest.json`` â€” any drift invalidates
comparisons across the campaign.
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Iterable

from shared.configs.schema import (
    AnalysisReport,
    BaselineMetrics,
    ExperimentResult,
    MetricsResult,
)


LOGGER = logging.getLogger("baseline_source.analyzer")


class Analyzer:
    """Parse, cross-check and aggregate per-seed metrics."""

    def __init__(self, campaign_dir: str | Path) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.campaign_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Parse a single metrics.json
    # ------------------------------------------------------------------
    def parse_metrics(self, metrics_path: str | Path) -> ExperimentResult:
        """Build an :class:`ExperimentResult` from a metrics.json file.

        ``arch_name`` / ``loss_name`` / ``arch_kwargs`` / ``loss_kwargs`` are
        read directly from the metrics payload (fix #3). The previous
        underscore-splitting parser corrupted identities for experiment IDs
        like ``grid_unet_v3_7level_nc16_masked_l1_s1``.
        """
        path = Path(metrics_path)
        m = MetricsResult.model_validate(json.loads(path.read_text()))

        holdout_r2 = (
            m.holdout_metrics.r2_median
            if m.holdout_metrics is not None else None
        )
        return ExperimentResult(
            experiment_id=m.experiment_id,
            strategy=m.strategy,
            arch_name=m.arch_name,
            loss_name=m.loss_name,
            arch_kwargs=dict(m.arch_kwargs or {}),
            loss_kwargs=dict(m.loss_kwargs or {}),
            seed=m.seed,
            val_r2_median=m.val_metrics.r2_median,
            val_r2_mean=m.val_metrics.r2_mean,
            holdout_r2_median=holdout_r2,
            wall_time_sec=m.wall_time_sec,
            peak_vram_gb=m.peak_vram_gb,
            status=m.status,
            metrics_path=str(path),
            script_path=m.script_path,
        )

    # ------------------------------------------------------------------
    # Hash cross-check
    # ------------------------------------------------------------------
    @staticmethod
    def cross_check_hashes(metrics_paths: Iterable[Path]) -> dict:
        eval_hashes: set[str] = set()
        split_hashes: set[str] = set()
        for p in metrics_paths:
            data = json.loads(Path(p).read_text())
            eval_hashes.add(data.get("eval_hash", ""))
            split_hashes.add(data.get("split_hash", ""))
        return {
            "eval_hash_consistent": len(eval_hashes) <= 1,
            "split_hash_consistent": len(split_hashes) <= 1,
            "eval_hashes": sorted(eval_hashes),
            "split_hashes": sorted(split_hashes),
        }

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def analyze(
        self,
        results: list[ExperimentResult],
        baseline: BaselineMetrics,
    ) -> list[AnalysisReport]:
        groups: dict[str, list[ExperimentResult]] = {}
        for r in results:
            groups.setdefault(self._group_key(r), []).append(r)

        reports: list[AnalysisReport] = []
        for key, entries in groups.items():
            vals = [
                e.val_r2_median for e in entries
                if e.status == "ok" and e.val_r2_median is not None
            ]
            if not vals:
                continue
            stats = {
                "mean": float(statistics.mean(vals)),
                "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
                "min": float(min(vals)),
                "max": float(max(vals)),
            }
            improvement = 100.0 * (
                (stats["mean"] - baseline.r2_median)
                / max(abs(baseline.r2_median), 1e-8)
            )
            reports.append(AnalysisReport(
                experiment_id=key,
                arch_name=entries[0].arch_name,
                loss_name=entries[0].loss_name,
                seeds_completed=len(vals),
                val_r2_median_stats=stats,
                val_r2_median_values=vals,
                improvement_pct=improvement,
            ))

        reports.sort(key=lambda r: r.val_r2_median_stats["mean"], reverse=True)
        for i, r in enumerate(reports):
            r.rank = i + 1
        return reports

    @staticmethod
    def _group_key(r: ExperimentResult) -> str:
        parts = r.experiment_id.split("_")
        if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
            return "_".join(parts[:-1])
        return f"{r.arch_name}_{r.loss_name}"

    # ------------------------------------------------------------------
    # Disk outputs
    # ------------------------------------------------------------------
    def write_results(
        self,
        results: list[ExperimentResult],
        reports: list[AnalysisReport],
    ) -> None:
        tsv = self.campaign_dir / "results.tsv"
        jsonl = self.campaign_dir / "results.jsonl"
        analysis_path = self.campaign_dir / "analysis.json"

        header_needed = not tsv.exists()
        with open(tsv, "a") as fh:
            if header_needed:
                fh.write(
                    "experiment_id\tstrategy\tarch\tloss\tseed"
                    "\tval_r2_median\tstatus\n"
                )
            for r in results:
                r2 = (
                    f"{r.val_r2_median:.6f}"
                    if r.val_r2_median is not None else "nan"
                )
                fh.write(
                    f"{r.experiment_id}\t{r.strategy}\t{r.arch_name}"
                    f"\t{r.loss_name}\t{r.seed}\t{r2}\t{r.status}\n"
                )

        with open(jsonl, "a") as fh:
            for r in results:
                fh.write(r.model_dump_json() + "\n")

        analysis_path.write_text(
            json.dumps([r.model_dump() for r in reports], indent=2)
        )


__all__ = ["Analyzer"]



