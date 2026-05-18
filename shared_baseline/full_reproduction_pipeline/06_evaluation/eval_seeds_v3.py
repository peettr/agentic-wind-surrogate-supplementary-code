#!/usr/bin/env python3
"""
Evaluate completed seeds in RAW wind speed domain.

Key insight: 878 patches map to 512 unique cases via patch_to_case.
Each case can have 1-9 patches. To restore predictions to raw domain,
we must pass ALL patches of a case to DataFormatter.restore_raw_output_data()
in the same order as DataFormatter originally produced them.

For each seed:
  1. Predict all 878 patches
  2. Group patches by case using patch_to_case
  3. For each test case, collect its patches, restore via DataFormatter
  4. Compute RÂ² in raw wind speed domain
  5. Compare with Lu's per-case RÂ²
"""
import os, sys, re, json, numpy as np, torch
import pandas as pd, xarray as xr
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

ROOT = Path('<PROJECT_HPC_ROOT>/auto_v2/full_dataset')
RAW_DIR = Path('<PROJECT_HPC_ROOT>/data/urbantales/raw')
REF_DIR = ROOT / 'references' / 'all_cases_20exp'

sys.path.insert(0, str(ROOT / 'scripts' / 'models'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'references'))
from unet_lu_7level import UNetLu7Level
from data_formatter_fixed import DataFormatterFixed as DataFormatter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
print(f"Device: {device}, GPU: {gpu_name}")

# =====================================================================
# Step 1: Load ALL formatted patches (878 total)
# =====================================================================
train_pt = torch.load(ROOT / 'data/full_masked_640/train.pt',
                      map_location='cpu', weights_only=False)
test_pt = torch.load(ROOT / 'data/full_masked_640/test.pt',
                     map_location='cpu', weights_only=False)
X_all = torch.cat([train_pt["X"], test_pt["X"]], dim=0)  # (878, 1, 640, 640)

# =====================================================================
# Step 2: Build correct per-patch -> case_name mapping
# train_pt: 705 patches, 410 unique cases, patch_to_case -> 0..409
# test_pt:  173 patches, 102 unique cases, patch_to_case -> 0..101
# Concatenated: 878 patches, 512 unique cases
# =====================================================================
train_cn = train_pt["case_names"]    # 410 names
test_cn = test_pt["case_names"]      # 102 names
train_p2c = train_pt["patch_to_case"]  # (705,)
test_p2c = test_pt["patch_to_case"]    # (173,)

patch_case_names = []  # length 878
for i in range(len(train_p2c)):
    patch_case_names.append(train_cn[int(train_p2c[i])])
for i in range(len(test_p2c)):
    patch_case_names.append(test_cn[int(test_p2c[i])])

assert len(patch_case_names) == X_all.shape[0], \
    f"Mapping error: {len(patch_case_names)} != {X_all.shape[0]}"

# Verify X has no NaN
assert not torch.isnan(X_all).any(), "X_all contains NaN!"

# case_name -> [patch_indices_in_X_all]
# Patch order is preserved: first train_pt patches in order, then test_pt
# This matches DataFormatter's internal patch ordering for each case
case_to_patches = {}
for idx, cn in enumerate(patch_case_names):
    if cn not in case_to_patches:
        case_to_patches[cn] = []
    case_to_patches[cn].append(idx)

print(f"Total patches: {X_all.shape[0]}, Unique cases: {len(case_to_patches)}")
del train_pt, test_pt

# =====================================================================
# Step 3: Load raw data for ALL cases
# =====================================================================
print("Loading raw data for all cases...")
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

# Verify patch counts match DataFormatter expectations
print("Verifying patch counts vs DataFormatter...")
n_mismatch = 0
for cn in sorted(case_to_patches.keys()):
    if cn not in all_raw:
        continue
    combined, angle = all_raw[cn]
    actual_patches = len(case_to_patches[cn])
    formatter = DataFormatter(raw_data=[combined], wind_angles=[angle],
                              formatted_shape=640)
    expected_patches = formatter._fmt_input_data.shape[0]
    if actual_patches != expected_patches:
        print(f"  MISMATCH: {cn} actual={actual_patches} expected={expected_patches}")
        n_mismatch += 1
print(f"Patch count verification: {n_mismatch} mismatches "
      f"out of {len(case_to_patches)} cases")


# =====================================================================
# Helper functions
# =====================================================================
def get_test_cases(seed):
    """Get set of test case names for this seed from Lu's CSV."""
    lu_test = pd.read_csv(REF_DIR / f"metrics_in_test_set_seed{seed}.csv")
    lu_test = lu_test[lu_test['topo'] != 'total']
    return set(lu_test['topo'].astype(str) + '_d' +
               lu_test['angle'].astype(str).str.zfill(2))


def get_lu_metrics(seed):
    """Get Lu's per-case RÂ² and global RÂ²."""
    lu_test = pd.read_csv(REF_DIR / f"metrics_in_test_set_seed{seed}.csv")
    total = lu_test[lu_test['topo'] == 'total']
    lu_global = total['r2_score'].values[0] if len(total) > 0 else float('nan')
    lu_test = lu_test[lu_test['topo'] != 'total']
    lu_r2 = {}
    lu_mae = {}
    for _, row in lu_test.iterrows():
        cn = f"{row['topo']}_d{str(int(row['angle'])).zfill(2)}"
        lu_r2[cn] = row['r2_score']
        lu_mae[cn] = row['mae']
    return lu_r2, lu_mae, lu_global


def predict_all(model_path):
    """Predict all 878 patches."""
    model = UNetLu7Level(n_c=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.eval()
    loader = DataLoader(TensorDataset(X_all.to(device)),
                        batch_size=4, shuffle=False)
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


def restore_and_r2(cn, pred_all):
    """
    For a single case:
      1. Collect ALL its patches from pred_all (in DataFormatter order)
      2. Pass to DataFormatter.restore_raw_output_data()
      3. Average restored patches (nanmean handles NaN in non-overlap regions)
      4. Compute RÂ² vs raw truth

    restore_raw_output_data returns (N, H_raw, W_raw) where N = number of
    patches, H_raw/W_raw = original dimensions. Each patch restores to the
    full raw domain with NaN in regions not covered by that patch.
    nanmean(axis=0) correctly blends overlapping regions.

    Returns: (r2, mae, truth_valid_array, pred_valid_array) or (nan,...,None,None)
    """
    if cn not in all_raw or cn not in case_to_patches:
        return float('nan'), float('nan'), None, None

    combined, angle = all_raw[cn]
    patch_indices = case_to_patches[cn]
    n_patches = len(patch_indices)

    # Create single-case DataFormatter
    formatter = DataFormatter(raw_data=[combined], wind_angles=[angle],
                              formatted_shape=640)
    expected_n = formatter._fmt_input_data.shape[0]

    if n_patches != expected_n:
        print(f"  ERROR: {cn} has {n_patches} patches but "
              f"DataFormatter expects {expected_n} â€” SKIPPING")
        return float('nan'), float('nan'), None, None

    # Collect predictions IN ORDER (matches DataFormatter's patch order)
    pred_patches = np.stack([pred_all[i] for i in patch_indices], axis=0)
    # Shape: (N, 640, 640) -> (N, 1, 640, 640)
    pred_input = pred_patches[:, np.newaxis, :, :]

    # Restore to raw domain
    # Returns (N, H_raw, W_raw) â€” each patch restored to full domain
    restored = formatter.restore_raw_output_data(pred_input)

    # Average across patches (NaN in non-covered regions â†’ nanmean blends)
    pred_raw = np.nanmean(restored, axis=0)
    truth = combined

    # Handle shape mismatch (should NOT happen â€” log warning if it does)
    if truth.shape != pred_raw.shape:
        if truth.shape == pred_raw.T.shape:
            print(f"  WARNING: {cn} needed transpose â€” "
                  f"investigate rotation bug!")
            pred_raw = pred_raw.T
        else:
            print(f"  ERROR: {cn} shape mismatch: "
                  f"truth={truth.shape}, pred={pred_raw.shape}")
            return float('nan'), float('nan'), None, None

    # RÂ² on valid (non-building, finite) pixels
    # Buildings encoded as negative values in combined: truth < 0 â†’ building
    valid = (truth >= 0) & np.isfinite(truth) & np.isfinite(pred_raw)
    t = truth[valid]
    p = pred_raw[valid]

    if len(t) == 0:
        return float('nan'), float('nan'), None, None

    mae = float(np.mean(np.abs(t - p)))
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return r2, mae, t, p


def eval_seed(seed, pred_np):
    """Evaluate one seed in raw domain and compare with Lu."""
    res_dir = ROOT / f'results/full_masked_640_7level/seed_{seed}'
    model_path = res_dir / 'model_best.pt'
    if not model_path.exists():
        print(f"Seed {seed}: no model_best.pt")
        return None

    test_cases = get_test_cases(seed)
    lu_r2_dict, lu_mae_dict, lu_global_r2 = get_lu_metrics(seed)

    print(f"\nSeed {seed}: {len(test_cases)} test cases from Lu")

    # Per-case RÂ² in raw domain
    case_r2 = {}
    case_mae = {}
    all_t, all_p = [], []
    n_ok = 0
    n_fail = 0

    for cn in sorted(test_cases):
        r2, mae, t, p = restore_and_r2(cn, pred_np)
        if not np.isnan(r2) and t is not None:
            case_r2[cn] = r2
            case_mae[cn] = mae
            all_t.append(t)
            all_p.append(p)
            n_ok += 1
        else:
            n_fail += 1

    if n_fail > 0:
        print(f"  WARNING: {n_fail} cases failed restore â€” "
              f"global RÂ² computed on {n_ok}/{len(test_cases)} cases only")

    # Global RÂ²
    all_t = np.concatenate(all_t)
    all_p = np.concatenate(all_p)
    global_r2 = 1 - np.sum((all_t - all_p)**2) / np.sum((all_t - all_t.mean())**2)
    global_mae = float(np.mean(np.abs(all_t - all_p)))

    # Compare with Lu
    common = sorted(set(case_r2.keys()) & set(lu_r2_dict.keys()))
    our_list = [case_r2[c] for c in common]
    lu_list = [lu_r2_dict[c] for c in common]
    wins = sum(1 for o, l in zip(our_list, lu_list) if o > l)
    losses = sum(1 for o, l in zip(our_list, lu_list) if o < l)

    print(f"  RAW Global RÂ² = {global_r2:.4f}, MAE = {global_mae:.4f}")
    print(f"  Lu Global RÂ² = {lu_global_r2:.4f}")
    print(f"  Per-case: {len(common)} matched, {wins}W/{losses}L")
    print(f"  Median: ours={np.nanmedian(our_list):.4f}, "
          f"Lu={np.nanmedian(lu_list):.4f}")
    print(f"  Mean: ours={np.nanmean(our_list):.4f}, "
          f"Lu={np.nanmean(lu_list):.4f}")

    return {
        'seed': seed,
        'global_r2': global_r2,
        'global_mae': global_mae,
        'lu_global_r2': lu_global_r2,
        'median_r2': np.nanmedian(our_list),
        'lu_median_r2': np.nanmedian(lu_list),
        'mean_r2': np.nanmean(our_list),
        'lu_mean_r2': np.nanmean(lu_list),
        'wins': wins,
        'losses': losses,
        'n_cases': len(common),
        'n_fail': n_fail,
    }


# =====================================================================
# Main
# =====================================================================
seeds = list(range(1, 21))
print("=" * 80)
print(f"{'Seed':>5} | {'Our RÂ²':>8} | {'Lu RÂ²':>8} | {'Î”':>8} | "
      f"{'W/L':>8} | {'Med':>8} | {'Lu Med':>8}")
print("-" * 80)

results = []
for s in seeds:
    res_dir = ROOT / f'results/full_masked_640_7level/seed_{s}'
    if not (res_dir / 'model_best.pt').exists():
        print(f"Seed {s}: skipping")
        continue
    pred_np = predict_all(res_dir / 'model_best.pt')
    r = eval_seed(s, pred_np)
    if r:
        results.append(r)
        d = r['global_r2'] - r['lu_global_r2']
        print(f"{r['seed']:5d} | {r['global_r2']:8.4f} | "
              f"{r['lu_global_r2']:8.4f} | {d:+8.4f} | "
              f"{r['wins']:3d}/{r['losses']:3d} | "
              f"{r['median_r2']:8.4f} | {r['lu_median_r2']:8.4f}")

print("=" * 80)
if results:
    a = np.mean([r['global_r2'] for r in results])
    b = np.mean([r['lu_global_r2'] for r in results])
    sa = np.std([r['global_r2'] for r in results])
    sb = np.std([r['lu_global_r2'] for r in results])
    print(f"Mean: Our={a:.4f}Â±{sa:.4f}, Lu={b:.4f}Â±{sb:.4f}, Î”={a-b:+.4f}")
    wins_total = sum(r['wins'] for r in results)
    losses_total = sum(r['losses'] for r in results)
    print(f"Total per-case W/L: {wins_total}/{losses_total}")
    print(f"Seeds winning Lu: {sum(1 for r in results if r['global_r2'] > r['lu_global_r2'])}/20")



