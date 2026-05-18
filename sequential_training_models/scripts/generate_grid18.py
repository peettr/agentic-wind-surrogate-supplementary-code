"""Generate 18 train_config.json files + a Condor submit file for the Grid 18 campaign.

Uses the Taguchi L18 orthogonal array from ``shared/search_space_builder.py`` to
cover n_c x loss x lr plus five folded 2-level factors (scheduler,
weight_decay, grad_clip, ema_decay, data_augment).

Architecture is fixed to ``unet_v2_baseline``. Three factors (activation,
dropout, norm_type) are NOT swept because UNetV2Baseline hardcodes
ReLU + BatchNorm + no-dropout.

Run:

    python3 generate_grid18.py

Writes:
    auto_v3/campaigns/grid18/run_00/train_config.json
    ...
    auto_v3/campaigns/grid18/run_17/train_config.json
    auto_v3/campaigns/grid18/grid18.submit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_AUTO_V3 = _HERE.parent
if str(_AUTO_V3) not in sys.path:
    sys.path.insert(0, str(_AUTO_V3))

# L18 orthogonal array (from search_space_builder.py, inlined to avoid yaml dependency)
# Each row: (nc_idx, _unused_arch_idx, loss_idx, lr_idx)
L18 = [
    (0, 0, 0, 0),
    (0, 0, 1, 1),
    (0, 0, 2, 2),
    (0, 1, 0, 1),
    (0, 1, 1, 2),
    (0, 1, 2, 0),
    (0, 2, 0, 2),
    (0, 2, 1, 0),
    (0, 2, 2, 1),
    (1, 0, 0, 2),
    (1, 0, 1, 0),
    (1, 0, 2, 1),
    (1, 1, 0, 0),
    (1, 1, 1, 1),
    (1, 1, 2, 2),
    (1, 2, 0, 1),
    (1, 2, 1, 2),
    (1, 2, 2, 0),
]


CRC_PREFIX = "<BASELINE_HPC_SOURCE_ROOT>"
CAMPAIGN_NAME = "grid18"

N_C_LEVELS = [16, 32]
LOSS_LEVELS = ["masked_l1", "masked_l1_gradient", "masked_huber"]
LR_LEVELS = [5e-4, 1e-3, 2e-3]

# 5 two-level factors — divisors 1..5 give 9/9 balance across 18 rows
SCHEDULER_LEVELS = [None, "cosine"]
WEIGHT_DECAY_LEVELS = [0, 1e-4]
GRAD_CLIP_LEVELS = [None, 0.5]
EMA_DECAY_LEVELS = [None, 0.999]
DATA_AUGMENT_LEVELS = [False, True]


def _fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _build_run(row_idx: int, row: tuple[int, int, int, int]) -> dict:
    nc_idx, _arch_idx, loss_idx, lr_idx = row

    n_c = N_C_LEVELS[nc_idx]
    loss = LOSS_LEVELS[loss_idx]
    lr = LR_LEVELS[lr_idx]

    # 5 two-level factors with divisors 1-5 (each appears 9 times in 18 rows)
    scheduler = SCHEDULER_LEVELS[(row_idx // 1) % 2]
    weight_decay = WEIGHT_DECAY_LEVELS[(row_idx // 2) % 2]
    grad_clip = GRAD_CLIP_LEVELS[(row_idx // 3) % 2]
    ema_decay = EMA_DECAY_LEVELS[(row_idx // 4) % 2]
    data_augment = DATA_AUGMENT_LEVELS[(row_idx // 5) % 2]

    experiment_id = f"grid18_{loss}_{_fmt_lr(lr)}_nc{n_c}_{row_idx:02d}"
    results_dir = f"{CRC_PREFIX}/campaigns/{CAMPAIGN_NAME}/run_{row_idx:02d}"

    training = {
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "ema_decay": ema_decay,
        "data_augment": data_augment,
    }

    return {
        "experiment_id": experiment_id,
        "strategy": "grid18",
        "seed": 1,
        "epochs": 150,
        "lr": lr,
        "batch_size": 16,
        "checkpoint_interval": 50,
        "arch_name": "unet_v2_baseline",
        "arch_kwargs": {"n_c": n_c, "training": training},
        "loss_name": loss,
        "loss_kwargs": {},
        "data_dir": f"{CRC_PREFIX}/shared/data",
        "results_dir": results_dir,
        "split_manifest_path": f"{CRC_PREFIX}/shared/data/split_manifest.json",
        "heartbeat_interval_epochs": 10,
        "phase": "search",
        "eval_splits": ["val"],
    }


def _write_submit(out_path: Path, n_runs: int) -> None:
    exe = f"{CRC_PREFIX}/templates/condor_wrapper.sh"
    train_py = f"{CRC_PREFIX}/shared/train.py"
    run_dir = f"{CRC_PREFIX}/campaigns/{CAMPAIGN_NAME}/run_$(Process)"
    log_dir = f"{CRC_PREFIX}/campaigns/{CAMPAIGN_NAME}"

    submit = (
        "universe = vanilla\n"
        "getenv = true\n"
        f"executable = {exe}\n"
        f'arguments = "{train_py} --config {run_dir}/train_config.json"\n'
        f"output = {run_dir}/condor.out\n"
        f"error = {run_dir}/condor.err\n"
        f"log = {log_dir}/condor.log\n"
        "request_gpus = 1\n"
        "request_memory = 16 GB\n"
        "requirements = GPUs_GlobalMemoryMb >= 40000\n"
        "should_transfer_files = NO\n"
        f"queue {n_runs}\n"
    )
    out_path.write_text(submit)


def _print_summary(runs: list[dict]) -> None:
    header = (
        f"{'#':>3}  {'n_c':>4}  {'loss':<22}  {'lr':>7}  "
        f"{'sched':<7}  {'wd':>6}  {'clip':>5}  "
        f"{'ema':<6}  {'aug':<5}"
    )
    print(header)
    print("-" * len(header))
    for i, cfg in enumerate(runs):
        t = cfg["arch_kwargs"]["training"]
        print(
            f"{i:>3}  {cfg['arch_kwargs']['n_c']:>4}  {cfg['loss_name']:<22}  "
            f"{cfg['lr']:>7.0e}  {str(t['scheduler']):<7}  "
            f"{t['weight_decay']:>6}  {str(t['grad_clip']):>5}  "
            f"{str(t['ema_decay']):<6}  {str(t['data_augment']):<5}"
        )


def main() -> None:
    campaign_dir = _AUTO_V3 / "campaigns" / CAMPAIGN_NAME
    campaign_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for row_idx, row in enumerate(L18):
        cfg = _build_run(row_idx, row)
        run_dir = campaign_dir / f"run_{row_idx:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "train_config.json").write_text(
            json.dumps(cfg, indent=2) + "\n"
        )
        runs.append(cfg)

    submit_path = campaign_dir / f"{CAMPAIGN_NAME}.submit"
    _write_submit(submit_path, n_runs=len(L18))

    _print_summary(runs)
    print()
    print(f"Wrote {len(runs)} train_config.json files to {campaign_dir}")
    print(f"Wrote submit file {submit_path}")


if __name__ == "__main__":
    main()
