#!/usr/bin/env python3
"""Regenerate Grid 18 + baseline configs using ORIGINAL L18 logic from generate_grid18.py.
Changes from original: epochs=200, compute_r2=true, heartbeat_interval=50."""
import json, os

BASE = "<BASELINE_HPC_SOURCE_ROOT>"
DIR = f"{BASE}/campaigns/grid18_200ep"

# ORIGINAL L18 table from generate_grid18.py
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

N_C_LEVELS = [16, 32]
LOSS_LEVELS = ["masked_l1", "masked_l1_gradient", "masked_huber"]
LR_LEVELS = [5e-4, 1e-3, 2e-3]
SCHEDULER_LEVELS = [None, "cosine"]
WEIGHT_DECAY_LEVELS = [0, 1e-4]
GRAD_CLIP_LEVELS = [None, 0.5]
EMA_DECAY_LEVELS = [None, 0.999]
DATA_AUGMENT_LEVELS = [False, True]

for row_idx, row in enumerate(L18):
    nc_idx, _arch_idx, loss_idx, lr_idx = row
    n_c = N_C_LEVELS[nc_idx]
    loss = LOSS_LEVELS[loss_idx]
    lr = LR_LEVELS[lr_idx]

    # ORIGINAL divisor-based 2-level factor assignment
    scheduler = SCHEDULER_LEVELS[(row_idx // 1) % 2]
    weight_decay = WEIGHT_DECAY_LEVELS[(row_idx // 2) % 2]
    grad_clip = GRAD_CLIP_LEVELS[(row_idx // 3) % 2]
    ema_decay = EMA_DECAY_LEVELS[(row_idx // 4) % 2]
    data_augment = DATA_AUGMENT_LEVELS[(row_idx // 5) % 2]

    training = {
        "scheduler": scheduler,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "ema_decay": ema_decay,
        "data_augment": data_augment,
    }

    cfg = {
        "experiment_id": f"grid18_200ep_{row_idx:02d}",
        "strategy": "grid18",
        "seed": 1,
        "epochs": 200,
        "lr": lr,
        "batch_size": 16,
        "checkpoint_interval": 50,
        "arch_name": "unet_v2_baseline",
        "arch_kwargs": {"n_c": n_c, "training": training},
        "loss_name": loss,
        "loss_kwargs": {},
        "data_dir": f"{BASE}/shared/data",
        "results_dir": f"{DIR}/run_{row_idx:02d}",
        "split_manifest_path": f"{BASE}/shared/data/split_manifest.json",
        "heartbeat_interval_epochs": 50,
        "phase": "search",
        "eval_splits": ["val"],
        "compute_r2": True,
    }
    
    run_dir = f"{DIR}/run_{row_idx:02d}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/train_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"run_{row_idx:02d}: n_c={n_c} loss={loss} lr={lr} sched={scheduler} wd={weight_decay} clip={grad_clip} ema={ema_decay} aug={data_augment}")

# Baseline
bl_cfg = {
    "experiment_id": "grid18_200ep_baseline",
    "strategy": "grid18",
    "seed": 1,
    "epochs": 200,
    "lr": 1e-3,
    "batch_size": 16,
    "checkpoint_interval": 50,
    "arch_name": "unet_v2_baseline",
    "arch_kwargs": {"n_c": 16, "training": {}},
    "loss_name": "masked_l1",
    "loss_kwargs": {},
    "data_dir": f"{BASE}/shared/data",
    "results_dir": f"{DIR}/baseline",
    "split_manifest_path": f"{BASE}/shared/data/split_manifest.json",
    "heartbeat_interval_epochs": 50,
    "phase": "search",
    "eval_splits": ["val"],
    "compute_r2": True,
}
bl_dir = f"{DIR}/baseline"
os.makedirs(bl_dir, exist_ok=True)
with open(f"{bl_dir}/train_config.json", "w") as f:
    json.dump(bl_cfg, f, indent=2)
print(f"baseline: n_c=16 loss=masked_l1 lr=1e-3")

# Verify: compare with original configs
print("\n=== Verification: compare run_02 with original ===")
import json as j
orig = j.load(open(f"{BASE}/campaigns/grid18/run_02/train_config.json"))
new = j.load(open(f"{DIR}/run_02/train_config.json"))
for key in ["lr", "loss_name"]:
    match = orig[key] == new[key]
    print(f"  {key}: orig={orig[key]} new={new[key]} match={match}")
orig_t = orig["arch_kwargs"]["training"]
new_t = new["arch_kwargs"]["training"]
for k in orig_t:
    match = orig_t[k] == new_t[k]
    print(f"  training.{k}: orig={orig_t[k]} new={new_t[k]} match={match}")
