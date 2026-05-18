#!/usr/bin/env python3
"""Compare Grid 18 runs vs Baseline on val set (51 cases, seed=7)."""
import json, os, re, sys
import torch
import numpy as np

BASE = "<BASELINE_HPC_SOURCE_ROOT>"
DATA_DIR = f"{BASE}/shared/data"
GRID_DIR = f"{BASE}/campaigns/grid18"
BL_DIR = f"{BASE}/campaigns/baseline_150ep_l40s"

# Load data + manifest
data = torch.load(f"{DATA_DIR}/all_data.pt", map_location="cpu", weights_only=False)
with open(f"{DATA_DIR}/split_manifest.json") as f:
    manifest = json.load(f)

sp = manifest["seeds"]["1"]
train_set = set(sp["train"])
val_set = set(sp["val"])

# Map patches to cases
p2c = data["patch_to_case"]
case_names = data["case_names"]

def get_patches(target_set):
    indices = [i for i in range(len(p2c)) if case_names[int(p2c[i])] in target_set]
    idx_t = torch.tensor(indices, dtype=torch.long)
    return data["X"][idx_t], data["Y"][idx_t], [case_names[int(p2c[i])] for i in indices]

X_val, Y_val, val_cases = get_patches(val_set)
# Deduplicate cases
val_unique = sorted(set(val_cases))
print(f"Val: {len(val_unique)} unique cases, {len(X_val)} patches")

sys.path.insert(0, f"{BASE}/shared")
from models.unet_v2_baseline import UNetV2Baseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Device: {device} ({gpu})")

X_val = X_val.to(device)
Y_val = Y_val.to(device)

def evaluate(model, X, Y):
    """Per-case R² and MAE on val set."""
    model.eval()
    results = {}
    # Group patches by case
    case_indices = {}
    for i, cn in enumerate(val_cases):
        if cn not in case_indices:
            case_indices[cn] = []
        case_indices[cn].append(i)
    
    with torch.no_grad():
        for cn in sorted(case_indices.keys()):
            idx = case_indices[cn]
            x = X[idx]
            y = Y[idx]
            pred = torch.cat([model(x[i:i+1]) for i in range(len(x))], dim=0)
            
            # Valid mask: not NaN and not building
            mask = (~torch.isnan(y)) & (x <= 0)
            
            p_flat = pred[mask].cpu().numpy()
            t_flat = y[mask].cpu().numpy()
            
            if len(t_flat) < 10:
                continue
            
            ss_res = np.sum((p_flat - t_flat) ** 2)
            ss_tot = np.sum((t_flat - t_flat.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            mae = np.mean(np.abs(p_flat - t_flat))
            results[cn] = {"r2": r2, "mae": mae}
    return results

def load_model(run_dir):
    """Load model from a run directory."""
    with open(f"{run_dir}/train_config.json") as f:
        cfg = json.load(f)
    
    n_c = cfg.get("arch_kwargs", {}).get("n_c", 16)
    
    model = UNetV2Baseline(n_c=n_c)
    
    model_path = f"{run_dir}/model_best.pt"
    if not os.path.exists(model_path):
        return None
    
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    # Handle both wrapped and unwrapped state dicts
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    return model

# === Evaluate baseline ===
print("\n=== Baseline (L40S, 150ep) ===")
bl_model = load_model(BL_DIR)
if bl_model is None:
    print("ERROR: No baseline model found!")
    sys.exit(1)

bl_results = evaluate(bl_model, X_val, Y_val)
bl_r2s = [v["r2"] for v in bl_results.values()]
bl_maes = [v["mae"] for v in bl_results.values()]
print(f"R² median: {np.median(bl_r2s):.4f}, mean: {np.mean(bl_r2s):.4f}")
print(f"MAE median: {np.median(bl_maes):.4f}, mean: {np.mean(bl_maes):.4f}")
del bl_model
torch.cuda.empty_cache()

# === Evaluate Grid 18 ===
print("\n=== Grid 18 Runs ===")
all_results = {"baseline": bl_results}
summary = []

for i in range(18):
    run_dir = f"{GRID_DIR}/run_{i:02d}"
    if not os.path.exists(f"{run_dir}/model_best.pt"):
        print(f"run_{i:02d}: SKIPPED (no model)")
        continue
    
    model = load_model(run_dir)
    if model is None:
        print(f"run_{i:02d}: FAILED to load")
        continue
    
    with open(f"{run_dir}/train_config.json") as f:
        cfg = json.load(f)
    
    res = evaluate(model, X_val, Y_val)
    r2s = [v["r2"] for v in res.values()]
    maes = [v["mae"] for v in res.values()]
    
    loss_name = cfg.get("loss_name", "?")
    training = cfg.get("arch_kwargs", {}).get("training", {})
    n_c = cfg.get("arch_kwargs", {}).get("n_c", 16)
    
    row = {
        "run": i,
        "n_c": n_c,
        "loss": loss_name,
        "lr": cfg.get("lr"),
        "r2_median": float(np.median(r2s)),
        "r2_mean": float(np.mean(r2s)),
        "mae_median": float(np.median(maes)),
        "mae_mean": float(np.mean(maes)),
    }
    summary.append(row)
    all_results[f"run_{i:02d}"] = res
    
    print(f"run_{i:02d}: n_c={n_c} loss={loss_name} lr={cfg.get('lr')} "
          f"R²={np.median(r2s):.4f} MAE={np.median(maes):.4f}")
    
    del model
    torch.cuda.empty_cache()

# === Save results ===
out_path = f"{BASE}/campaigns/grid18_comparison.json"
with open(out_path, "w") as f:
    json.dump({"baseline": {"r2_median": float(np.median(bl_r2s)), "r2_mean": float(np.mean(bl_r2s)),
                            "mae_median": float(np.median(bl_maes)), "mae_mean": float(np.mean(bl_maes))},
               "grid18": summary}, f, indent=2)
print(f"\nResults saved to {out_path}")

# === Head-to-head comparison ===
print("\n=== Head-to-Head: each run vs baseline (per-case R² win/loss) ===")
for row in sorted(summary, key=lambda x: x["r2_median"], reverse=True):
    run_key = f"run_{row['run']:02d}"
    if run_key not in all_results:
        continue
    wins = 0
    losses = 0
    for cn in bl_results:
        if cn in all_results[run_key]:
            if all_results[run_key][cn]["r2"] > bl_results[cn]["r2"]:
                wins += 1
            else:
                losses += 1
    row["wins"] = wins
    row["losses"] = losses
    print(f"run_{row['run']:02d}: W={wins} L={losses} | R²={row['r2_median']:.4f} MAE={row['mae_median']:.4f} | n_c={row['n_c']} {row['loss']} lr={row['lr']}")

print(f"\nBaseline: R²={np.median(bl_r2s):.4f} MAE={np.median(bl_maes):.4f}")
