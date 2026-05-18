"""CLI entry point: run a Grid or AI Explorer campaign.

Usage::

    python scripts/run_campaign.py --strategy grid --campaign-dir campaigns/grid_run1
    python scripts/run_campaign.py --strategy ai_explorer --campaign-dir campaigns/ai_run1

All strategy-specific behaviour is carried by the planner object; the
runner itself is strategy-agnostic (see ``engine/runner.py``).
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.executor import Executor
from engine.runner import run_campaign
from shared.codegen_service import CodegenService
from shared.configs.schema import BaselineMetrics, ExperimentConfig
from shared.preflight import Preflight
from shared.review_engine import ReviewEngine
from shared.search_space_builder import SearchSpaceBuilder
from strategies.ai_explorer.planner import AIExplorerPlanner
from strategies.grid.planner import GridPlanner


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("baseline_source.cli")


def _default_baseline() -> BaselineMetrics:
    return BaselineMetrics(
        arch_name="unet_v2_baseline",
        loss_name="masked_l1",
        r2_median=0.7017,
        r2_std=0.005,
        seeds=[1, 7, 42],
        source="v2 7-level UNet (unet_lu_7level.py), 20 seeds",
    )


def _baseline_configs(
    baseline: BaselineMetrics, epochs: int = 500,
) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            experiment_id=(
                f"baseline_{baseline.arch_name}_{baseline.loss_name}_s{s}"
            ),
            strategy="baseline",
            arch_name=baseline.arch_name,
            loss_name=baseline.loss_name,
            variant={"arch_kwargs": {"n_c": 16}},
            seed=s,
            epochs=epochs,
            lr=1.0e-3,
            batch_size=16,
            n_c=16,
        )
        for s in baseline.seeds
    ]


def _noop_llm(prompt: str) -> list[dict]:
    LOGGER.warning(
        "llm_fn not configured; returning no proposals. "
        "Provide --llm-cli or wire your own callable."
    )
    return []


def _make_llm_fn(cli: str):
    exe = shutil.which(cli)
    if exe is None:
        LOGGER.warning(
            "LLM CLI '%s' not on PATH; proposals will be empty.", cli,
        )
        return _noop_llm

    def _call(prompt: str) -> list[dict]:
        try:
            res = subprocess.run(
                [exe, "--print", "-"], input=prompt,
                capture_output=True, text=True, timeout=600, check=False,
            )
            text = res.stdout or ""
            start, end = text.find("["), text.rfind("]")
            if start == -1 or end == -1:
                return []
            return json.loads(text[start : end + 1])
        except Exception as exc:
            LOGGER.exception("LLM invocation failed: %s", exc)
            return []

    return _call


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--strategy", choices=["grid", "ai_explorer"], required=True,
    )
    ap.add_argument("--campaign-dir", type=Path, required=True)
    ap.add_argument(
        "--problem-def", type=Path,
        default=_ROOT / "shared" / "configs" / "problem_definition.yaml",
    )
    ap.add_argument(
        "--search-space", type=Path,
        default=_ROOT / "shared" / "configs" / "search_space.json",
    )
    ap.add_argument(
        "--campaign-config", type=Path,
        help="Path to campaign_config.json; defaults to the strategy bundle.",
    )
    ap.add_argument(
        "--split-manifest", type=Path,
        default=_ROOT / "shared" / "data" / "split_manifest.json",
    )
    ap.add_argument(
        "--data-dir", type=Path, default=_ROOT / "shared" / "data",
    )
    ap.add_argument(
        "--llm-cli", type=str, default="claude",
        help="CLI executable for LLM proposals",
    )
    ap.add_argument("--baseline-epochs", type=int, default=500)
    args = ap.parse_args()

    baseline = _default_baseline()
    ssb = SearchSpaceBuilder(args.problem_def)
    space = ssb.get_search_space(args.search_space)
    reference_space = ssb.reference_space(args.search_space)

    cfg_path = (
        args.campaign_config
        or _ROOT / "strategies" / args.strategy / "campaign_config.json"
    )
    campaign_config = json.loads(Path(cfg_path).read_text())

    codegen = CodegenService(
        review_engine=ReviewEngine(), preflight=Preflight(),
    )
    llm_fn = _make_llm_fn(args.llm_cli)

    if args.strategy == "grid":
        planner = GridPlanner(
            search_space=space,
            campaign_config=campaign_config,
            codegen=codegen,
            reference_space=reference_space,
            llm_fn=llm_fn,
        )
    else:
        planner = AIExplorerPlanner(
            campaign_config=campaign_config,
            codegen=codegen,
            reference_space=reference_space,
            llm_fn=llm_fn,
        )

    executor = Executor(
        campaign_dir=args.campaign_dir,
        data_dir=args.data_dir,
        split_manifest_path=args.split_manifest,
    )

    champion = run_campaign(
        planner=planner,
        baseline=baseline,
        baseline_configs=_baseline_configs(baseline, args.baseline_epochs),
        campaign_dir=args.campaign_dir,
        strategy=args.strategy,
        executor=executor,
        split_manifest_path=args.split_manifest,
    )
    if champion:
        print(
            f"\nChampion: {champion.experiment_id}  "
            f"val_r2_median={champion.val_r2_median:.4f}"
        )
    else:
        print("\nNo champion selected (no successful runs).")


if __name__ == "__main__":
    main()



