#!/usr/bin/env python3
"""
Comprehensive evaluation: all metrics for 20 seeds (train + test).
Outputs: per_case_metrics_20seeds.json
Metrics per case: RÂ², MAE, NMAE, Îµ_Ïƒ, U_mean, U_std, U_max (pred & LES)
"""
import os, sys, re, json, numpy as np, torch
import pandas as pd, xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path('<PROJECT_HPC_ROOT>/auto_v2/full_dataset')
RAW_DIR = Path('<PROJECT_HPC_ROOT>/data/urbantales/raw')
REF_DIR = ROOT / 'references' / 'all_cases_20exp'

sys.path.insert(0, str(ROOT / 'scripts' / 'models'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'references'))
from unet_lu_7level import UNetLu7Level
from data_formatter_fixed import DataFormatterFixed as DataFormatter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, GPU: {torch.cuda.get_device_name(device)}")

# Load all patches
train_pt = torch.load(ROOT / 'data/full_masked_640/train.pt', map_location='cpu', weights_only=False)
test_pt = torch.load(ROOT / 'data/full_masked_640/test.pt', map_location='cpu', weights_only=False)
X_all = torch.cat([train_pt["X"], test_pt["X"]], dim=0)

train_cn = train_pt["case_names"]
test_cn = test_pt["case_names"]
train_p2c = train_pt["patch_to_case"]
test_p2c = test_pt["patch_to_case"]
n_train = len(train_p2c)

patch_case_names = []
for i in range(n_train):
    patch_case_names.append(train_cn[int(train_p2c[i])])
for i in range(len(test_p2c)):
    patch_case_names.append(test_cn[int(test_p2c[i])])

case_to_patches = {}
for idx, cn in enumerate(patch_case_names):
    if cn not in case_to_patches:
        case_to_patches[cn] = []
    case_to_patches[cn].append(idx)

del train_pt, test_pt

# Load raw data
print("Loading raw data...")
all_raw = {}
for cn in sorted(case_to_patches.keys()):
    case_dir = RAW_DIR / cn
    if not case_dir.is_dir():
        continue
    try:
        topo = np.flipud(np.loadtxt(case_dir / f"{cn}_topo", dtype=np.float32))
        with xr.open_dataset(case_dir / f"{cn}_ped.nc") as ds:
            Uped = ds["Uped"].values.astype(np.float32)
        combined = Uped.copy()
        combined[topo > 0] = -topo[topo > 0]
        m = re.search(r"_d(\d+)$", cn)
        angle = int(m.group(1))
        all_raw[cn] = (combined, angle)
    except Exception as e:
        print(f"  Failed: {cn}: {e}")
print(f"Loaded {len(all_raw)} raw cases")


def is_ideal(cn):
    p = cn.split('_')[0]
    return p[:2] in ('VA','VS','UA','US') and len(p)>2 and p[2].isdigit()


def get_density(cn):
    """Extract plan area density from case name (idealized only)."""
    p = cn.split('_')[0]
    d = p[2:]
    mapping = {'0625':0.0625, '1111':0.1111, '25':0.25, '4444':4/9, '64':0.64}
    return mapping.get(d, None)


def predict_all(model_path):
    model = UNetLu7Level(n_c=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    loader = DataLoader(TensorDataset(X_all.to(device)), batch_size=16, shuffle=False)
    all_pred = []
    with torch.no_grad():
        for (x,) in loader:
            all_pred.append(model(x))
    pred = np.nan_to_num(
        torch.cat(all_pred, dim=0).cpu().numpy()[:, 0, :, :],
        nan=0.0).astype(np.float32)
    del model
    torch.cuda.empty_cache()
    return pred


def compute_case_metrics(cn, pred_all):
    """Compute all metrics for a single case."""
    if cn not in all_raw or cn not in case_to_patches:
        return None
    combined, angle = all_raw[cn]
    patch_indices = case_to_patches[cn]
    n_patches = len(patch_indices)
    formatter = DataFormatter(raw_data=[combined], wind_angles=[angle], formatted_shape=640)
    expected_n = formatter._fmt_input_data.shape[0]
    if n_patches != expected_n:
        return None
    pred_patches = np.stack([pred_all[i] for i in patch_indices], axis=0)
    pred_input = pred_patches[:, np.newaxis, :, :]
    restored = formatter.restore_raw_output_data(pred_input)
    pred_raw = np.nanmean(restored, axis=0)
    truth = combined
    if truth.shape != pred_raw.shape:
        if truth.shape == pred_raw.T.shape:
            pred_raw = pred_raw.T
        else:
            return None
    valid = (truth >= 0) & np.isfinite(truth) & np.isfinite(pred_raw)
    t = truth[valid].astype(np.float64)
    p = pred_raw[valid].astype(np.float64)
    if len(t) == 0:
        return None
    ss_res = np.sum((t - p)**2)
    ss_tot = np.sum((t - t.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -999.0
    mae = float(np.mean(np.abs(t - p)))
    rmse = float(np.sqrt(np.mean((t - p)**2)))
    nmae = mae / np.mean(t) * 100 if np.mean(t) > 0 else -999.0
    std_t = np.std(t)
    std_p = np.std(p)
    eps_sigma = (std_p - std_t) / std_t * 100 if std_t > 0 else -999.0
    return {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'nmae': float(nmae),
        'eps_sigma': float(eps_sigma),
        'u_mean_pred': float(np.mean(p)),
        'u_mean_les': float(np.mean(t)),
        'u_std_pred': float(std_p),
        'u_std_les': float(std_t),
        'u_max_pred': float(np.max(p)),
        'u_max_les': float(np.max(t)),
        'n_pixels': int(len(t)),
    }


def get_seed_split(seed):
    """Get train/test case sets from Lu's CSV."""
    lu_test = pd.read_csv(REF_DIR / f"metrics_in_test_set_seed{seed}.csv")
    lu_test = lu_test[lu_test['topo'] != 'total']
    test_cases = set(lu_test['topo'].astype(str) + '_d' + lu_test['angle'].astype(str).str.zfill(2))
    lu_r2_map = {}
    for _, row in lu_test.iterrows():
        cn = f"{row['topo']}_d{str(int(row['angle'])).zfill(2)}"
        lu_r2_map[cn] = float(row['r2_score'])
    lu_train = pd.read_csv(REF_DIR / f"metrics_in_training_set_seed{seed}.csv")
    lu_train = lu_train[lu_train['topo'] != 'total']
    train_cases = set(lu_train['topo'].astype(str) + '_d' + lu_train['angle'].astype(str).str.zfill(2))
    return test_cases, train_cases, lu_r2_map


all_results = {}

for seed in range(1, 21):
    res_dir = ROOT / f'results/full_masked_640_7level/seed_{seed}'
    model_path = res_dir / 'model_best.pt'
    if not model_path.exists():
        print(f"Seed {seed}: missing model")
        continue
    print(f"\n=== Seed {seed} ===")
    pred_np = predict_all(model_path)
    test_cases, train_cases, lu_r2_map = get_seed_split(seed)

    # Test set
    test_per_case = []
    for cn in sorted(test_cases):
        m = compute_case_metrics(cn, pred_np)
        if m is None:
            continue
        m['case'] = cn
        m['lu_r2'] = lu_r2_map.get(cn, float('nan'))
        m['is_idealized'] = is_ideal(cn)
        m['density'] = get_density(cn)
        test_per_case.append(m)

    # Train set (just global stats, not per-case to save time)
    train_per_case = []
    for cn in sorted(train_cases):
        m = compute_case_metrics(cn, pred_np)
        if m is None:
            continue
        m['case'] = cn
        m['is_idealized'] = is_ideal(cn)
        m['density'] = get_density(cn)
        train_per_case.append(m)

    # Global test RÂ²
    all_t, all_p = [], []
    for m in test_per_case:
        cn = m['case']
        combined, angle = all_raw[cn]
        patch_indices = case_to_patches[cn]
        pred_patches = np.stack([pred_np[i] for i in patch_indices], axis=0)
        formatter = DataFormatter(raw_data=[combined], wind_angles=[angle], formatted_shape=640)
        pred_input = pred_patches[:, np.newaxis, :, :]
        restored = formatter.restore_raw_output_data(pred_input)
        pred_raw = np.nanmean(restored, axis=0)
        if combined.shape != pred_raw.shape:
            if combined.shape == pred_raw.T.shape:
                pred_raw = pred_raw.T
            else:
                continue
        valid = (combined >= 0) & np.isfinite(combined) & np.isfinite(pred_raw)
        all_t.append(combined[valid])
        all_p.append(pred_raw[valid])
    all_t = np.concatenate(all_t)
    all_p = np.concatenate(all_p)
    global_r2 = 1 - np.sum((all_t - all_p)**2) / np.sum((all_t - all_t.mean())**2)
    global_mae = float(np.mean(np.abs(all_t - all_p)))

    wins = sum(1 for e in test_per_case if e['r2'] > e.get('lu_r2', -999))
    losses = sum(1 for e in test_per_case if e['r2'] < e.get('lu_r2', 999))

    all_results[str(seed)] = {
        'global_r2': float(global_r2),
        'global_mae': float(global_mae),
        'test_per_case': test_per_case,
        'train_per_case': train_per_case,
        'wins': wins,
        'losses': losses,
    }
    print(f"  Global RÂ² = {global_r2:.4f}, MAE = {global_mae:.4f}, W/L = {wins}/{losses}")
    print(f"  Test: {len(test_per_case)} cases, Train: {len(train_per_case)} cases")
    del pred_np

out_path = ROOT / 'results' / 'per_case_metrics_20seeds.json'
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")



