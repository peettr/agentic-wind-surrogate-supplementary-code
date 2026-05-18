#!/usr/bin/env python3
"""Generate 19 configs for Grid 18 re-run + baseline, 200ep, compute_r2=true."""
import json, os

BASE = "<BASELINE_HPC_SOURCE_ROOT>"
DIR = f"{BASE}/campaigns/grid18_200ep"
os.makedirs(DIR, exist_ok=True)

# Grid 18 L18 table (from generate_grid18.py)
# 8 factors: loss(3), lr(3), n_c(2), scheduler(2), wd(2), grad_clip(2), ema(2), augment(2)
L18 = [
    # loss, lr, n_c, scheduler, wd, grad_clip, ema, augment
    (0, 0, 0, 0, 0, 0, 0, 0),  # run_00
    (1, 0, 1, 1, 1, 1, 1, 1),  # run_01
    (2, 0, 0, 1, 1, 1, 0, 1),  # run_02
    (0, 1, 1, 1, 0, 1, 1, 0),  # run_03
    (1, 1, 0, 0, 1, 0, 1, 1),  # run_04
    (2, 1, 1, 0, 0, 1, 0, 1),  # run_05
    (0, 2, 0, 1, 1, 0, 1, 0),  # run_06
    (1, 2, 1, 0, 0, 0, 0, 0),  # run_07
    (2, 2, 0, 0, 1, 1, 1, 0),  # run_08
    (0, 0, 1, 0, 1, 1, 1, 0),  # run_09
    (1, 0, 0, 1, 0, 0, 0, 1),  # run_10
    (2, 0, 1, 0, 0, 1, 0, 1),  # run_11
    (0, 1, 0, 0, 0, 0, 1, 1),  # run_12
    (1, 1, 1, 1, 1, 0, 0, 0),  # run_13
    (2, 1, 0, 1, 0, 1, 1, 1),  # run_14
    (0, 2, 1, 1, 0, 1, 0, 1),  # run_15
    (1, 2, 0, 0, 0, 1, 1, 0),  # run_16
    (2, 2, 1, 1, 1, 0, 0, 1),  # run_17
]

LOSSES = ["masked_l1", "masked_l1_gradient", "masked_huber"]
LRS = [5e-4, 1e-3, 2e-3]
N_CS = [16, 32]
SCHEDULERS = [None, "cosine"]
WDS = [0, 1e-4]
CLIPS = [None, 0.5]
EMAS = [None, 0.999]
AUGMENTS = [False, True]

common = {
    "seed": 1,
    "batch_size": 16,
    "checkpoint_interval": 50,
    "arch_name": "unet_v2_baseline",
    "loss_kwargs": {},
    "data_dir": f"{BASE}/shared/data",
    "split_manifest_path": f"{BASE}/shared/data/split_manifest.json",
    "heartbeat_interval_epochs": 50,
    "phase": "search",
    "strategy": "grid18",
    "eval_splits": ["val"],
    "compute_r2": True,
}

# Generate 18 grid configs
for i, (li, lri, nci, schi, wdi, cli, emi, aui) in enumerate(L18):
    n_c = N_CS[nci]
    # Check param limit: n_c=32 = 138M < 150M, ok
    loss_name = LOSSES[li]
    lr = LRS[lri]
    training = {}
    if SCHEDULERS[schi]:
        training["scheduler"] = SCHEDULERS[schi]
    if WDS[wdi]:
        training["weight_decay"] = WDS[wdi]
    if CLIPS[cli]:
        training["grad_clip"] = CLIPS[cli]
    if EMAS[emi]:
        training["ema_decay"] = EMAS[emi]
    if AUGMENTS[aui]:
        training["data_augment"] = AUGMENTS[aui]

    cfg = dict(common)
    cfg.update({
        "experiment_id": f"grid18_200ep_{i:02d}",
        "epochs": 200,
        "lr": lr,
        "loss_name": loss_name,
        "arch_kwargs": {"n_c": n_c, "training": training},
        "results_dir": f"{DIR}/run_{i:02d}",
    })
    
    run_dir = f"{DIR}/run_{i:02d}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/train_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"run_{i:02d}: n_c={n_c} loss={loss_name} lr={lr} train_extras={training}")

# Generate baseline config
bl_cfg = dict(common)
bl_cfg.update({
    "experiment_id": "grid18_200ep_baseline",
    "epochs": 200,
    "lr": 1e-3,
    "loss_name": "masked_l1",
    "arch_kwargs": {"n_c": 16, "training": {}},
    "results_dir": f"{DIR}/baseline",
})
bl_dir = f"{DIR}/baseline"
os.makedirs(bl_dir, exist_ok=True)
with open(f"{bl_dir}/train_config.json", "w") as f:
    json.dump(bl_cfg, f, indent=2)
print(f"baseline: n_c=16 loss=masked_l1 lr=1e-3")

# Generate Condor submit file
submit_lines = [
    f"executable = {BASE}/templates/condor_wrapper.sh",
    f"output = {DIR}/$(Name).out",
    f"error = {DIR}/$(Name).err",
    f"log = {DIR}/condor.log",
    "request_gpus = 1",
    "request_memory = 16 GB",
    "requirements = (GPUs_GlobalMemoryMb >= 40000)",
    "should_transfer_files = NO",
]

for i in range(18):
    submit_lines.append(f"\nName = run_{i:02d}")
    submit_lines.append(f'arguments = "{BASE}/shared/train.py --config {DIR}/run_{i:02d}/train_config.json"')
    submit_lines.append("queue 1")

submit_lines.append(f"\nName = baseline")
submit_lines.append(f'arguments = "{BASE}/shared/train.py --config {DIR}/baseline/train_config.json"')
submit_lines.append("queue 1")

with open(f"{DIR}/grid18_200ep.submit", "w") as f:
    f.write("\n".join(submit_lines) + "\n")

print(f"\nSubmit file: {DIR}/grid18_200ep.submit")
print("Total: 19 jobs (18 grid + 1 baseline)")
