"""Final validation: champion Ã— 20 seeds Ã— 1000 epochs + holdout eval.

Runs **only** after an explicit ``--approve`` flag. Produces a Lu-comparable
report in ``{campaign_dir}/final_validation/`` containing per-seed metrics
and a holdout evaluation via the locked :class:`EvalModule`.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.analyzer import Analyzer
from engine.executor import Executor
from shared.configs.schema import ExperimentConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("baseline_source.final")

FINAL_SEEDS = list(range(1, 21))
FINAL_EPOCHS = 1000


def load_champion(campaign_dir: Path) -> dict:
    """Return the champion entry from state.json + history.jsonl.

    state.json is now small (planner mode + champion snapshot), and the
    full history lives in history.jsonl (single source of truth). If the
    champion snapshot isn't embedded, scan history.jsonl for the matching
    experiment_id.
    """
    state_path = campaign_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"No campaign state at {state_path}")
    state = json.loads(state_path.read_text())
    champion_id = state.get("champion_id")
    if not champion_id:
        raise RuntimeError("Campaign has no champion recorded")
    champ_snap = state.get("champion")
    if champ_snap and champ_snap.get("experiment_id") == champion_id:
        return champ_snap
    hist_path = campaign_dir / "history.jsonl"
    if hist_path.exists():
        with open(hist_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("experiment_id") == champion_id:
                    return entry
    raise RuntimeError(f"Champion {champion_id} not in history")


def make_configs(champion: dict) -> list[ExperimentConfig]:
    """Recreate champion configs for 20-seed Ã— 1000-epoch final validation.

    Fix #9: preserve the champion's full ``arch_kwargs``/``loss_kwargs`` â€”
    retraining with an empty variant would silently run the default
    constructor and lose generated / non-default-depth UNet variants /
    FNO width/modes / custom loss kwargs.
    """
    arch_kwargs = dict(champion.get("arch_kwargs") or {})
    loss_kwargs = dict(champion.get("loss_kwargs") or {})
    n_c = int(arch_kwargs.get("n_c", 0))
    # Generated architectures/losses must preserve the script_path so
    # train.py can dynamically import the champion's module on every
    # final-validation seed (new issue 2 / partial 9).
    champion_script_path = champion.get("script_path")
    return [
        ExperimentConfig(
            experiment_id=(
                f"final_{champion['arch_name']}_{champion['loss_name']}_s{s}"
            ),
            strategy="final",
            arch_name=champion["arch_name"],
            loss_name=champion["loss_name"],
            variant={
                "arch_kwargs": arch_kwargs,
                "loss_kwargs": loss_kwargs,
            },
            seed=s,
            epochs=FINAL_EPOCHS,
            lr=1.0e-3,
            batch_size=16,
            n_c=n_c,
            phase="final",
            script_path=champion_script_path,
        )
        for s in FINAL_SEEDS
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", type=Path, required=True)
    ap.add_argument(
        "--data-dir", type=Path, default=_ROOT / "shared" / "data",
    )
    ap.add_argument(
        "--split-manifest", type=Path,
        default=_ROOT / "shared" / "data" / "split_manifest.json",
    )
    ap.add_argument(
        "--approve", action="store_true",
        help="Explicit confirmation to run 20-seed x 1000-epoch final.",
    )
    args = ap.parse_args()

    if not args.approve:
        print("ERROR: Pass --approve to run the 20-seed final validation.")
        sys.exit(2)

    champion = load_champion(args.campaign_dir)
    LOGGER.info("Champion: %s", champion["experiment_id"])

    out_dir = args.campaign_dir / "final_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = Executor(
        campaign_dir=out_dir,
        data_dir=args.data_dir,
        split_manifest_path=args.split_manifest,
    )
    handles = executor.submit_batch(make_configs(champion))
    raw = executor.collect_results(handles)

    # Per-seed summary -------------------------------------------------
    results = []
    for m in raw:
        val = m.get("val_metrics") or {}
        holdout = m.get("holdout_metrics") or {}
        results.append({
            "experiment_id": m.get("experiment_id"),
            "seed": m.get("seed"),
            "val_r2_median": val.get("r2_median"),
            "holdout_r2_median": holdout.get("r2_median"),
            "wall_time_sec": m.get("wall_time_sec"),
            "status": m.get("status"),
        })
    (out_dir / "final_results.json").write_text(json.dumps(results, indent=2))

    # Aggregate + Lu comparison ---------------------------------------
    # Filter by status == "ok" and finite metrics: the executor's failure
    # placeholder uses NaN values which JSON accepts, so without this
    # guard failed seeds would inflate seeds_completed.
    val_r2 = [
        r["val_r2_median"] for r in results
        if r.get("status") == "ok"
        and r["val_r2_median"] is not None
        and not math.isnan(r["val_r2_median"])
    ]
    hold_r2 = [
        r["holdout_r2_median"] for r in results
        if r.get("status") == "ok"
        and r["holdout_r2_median"] is not None
        and not math.isnan(r["holdout_r2_median"])
    ]
    report = {
        "champion_id": champion["experiment_id"],
        "seeds_completed": len(val_r2),
        "val_r2_median_across_seeds": (
            statistics.median(val_r2) if val_r2 else None
        ),
        "val_r2_mean_across_seeds": (
            statistics.mean(val_r2) if val_r2 else None
        ),
        "val_r2_std_across_seeds": (
            statistics.pstdev(val_r2) if len(val_r2) > 1 else 0.0
        ),
        "holdout_r2_median_across_seeds": (
            statistics.median(hold_r2) if hold_r2 else None
        ),
        "baseline_lu_r2_median": 0.7017,
    }
    (out_dir / "lu_comparison.json").write_text(json.dumps(report, indent=2))

    # Optional rich aggregation via analyzer (writes analysis.json)
    _ = Analyzer(out_dir)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()



