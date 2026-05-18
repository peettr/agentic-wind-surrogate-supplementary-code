#!/usr/bin/env python3
"""Re-evaluate baseline model_best.pt on the new seed=7 val split."""
import sys, json, torch, numpy as np
sys.path.insert(0, "<BASELINE_HPC_SOURCE_ROOT>")

from shared.models import REGISTRY
from shared.eval_module import EvalModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Device: {device}, GPU: {gpu}")

# Load baseline model
model_cls = REGISTRY.get("unet_v2_baseline")
model = model_cls(n_c=16)
ckpt = torch.load(
    "<BASELINE_HPC_SOURCE_ROOT>/campaigns/baseline/runs/baseline_s1/model_best.pt",
    map_location="cpu", weights_only=True
)
model.load_state_dict(ckpt)
model = model.to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Model params: {params:,}")

# Eval on new val split (seed=7 manifest)
manifest_path = "<BASELINE_HPC_SOURCE_ROOT>/shared/data/split_manifest.json"
data_dir = "<BASELINE_HPC_SOURCE_ROOT>/shared/data"

ev = EvalModule(split_manifest_path=manifest_path, data_dir=data_dir)
val_result = ev.evaluate(model, split="val", seed=1, batch_size=4, device=device)

print("\n=== Baseline on new Val (seed=7) ===")
print(f"Global MAE: {val_result.get('global_mae', 'N/A')}")
print(f"Global R2:  {val_result.get('global_r2', 'N/A')}")

per_case_r2 = val_result.get("per_case_r2", {})
per_case_mae = val_result.get("per_case_mae", {})
if per_case_r2:
    r2_vals = list(per_case_r2.values())
    mae_vals = list(per_case_mae.values())
    print(f"Per-case R2 median:  {np.median(r2_vals):.4f}")
    print(f"Per-case MAE median: {np.median(mae_vals):.4f}")
    print(f"Per-case R2 mean:    {np.mean(r2_vals):.4f}")
    print(f"Per-case MAE mean:   {np.mean(mae_vals):.4f}")
    print(f"Num cases: {len(r2_vals)}")
