"""Concatenate v2's train.pt + test.pt into v3's all_data.pt format.

Uses v2's DataFormatter-generated data (878 patches, 512 cases) so that
training and eval are byte-identical to v2.

Usage (on CRC login node with graphwind conda):
    python concat_v2_data.py
"""
import re
import torch
import numpy as np
import xarray as xr
from pathlib import Path

V2_DIR = Path("<PROJECT_HPC_ROOT>/auto_v2/full_dataset/data/full_masked_640")
V3_DIR = Path("<BASELINE_HPC_SOURCE_ROOT>/shared/data")
RAW_DIR = Path("<PROJECT_HPC_ROOT>/data/urbantales/raw")

print("Loading v2 data...")
train_pt = torch.load(V2_DIR / "train.pt", map_location="cpu", weights_only=False)
test_pt = torch.load(V2_DIR / "test.pt", map_location="cpu", weights_only=False)

# v2 format: X, Y, nan_mask, case_names, wind_angles, patch_to_case, fmt_shape
print(f"  train: X={train_pt['X'].shape} cases={len(train_pt['case_names'])} patches={len(train_pt['patch_to_case'])}")
print(f"  test:  X={test_pt['X'].shape} cases={len(test_pt['case_names'])} patches={len(test_pt['patch_to_case'])}")

# Concatenate tensors
X = torch.cat([train_pt["X"], test_pt["X"]], dim=0)
Y = torch.cat([train_pt["Y"], test_pt["Y"]], dim=0)
nan_mask = torch.cat([train_pt["nan_mask"], test_pt["nan_mask"]], dim=0)

# Merge case_names: train's 410 + test's 102 = 512 unique
train_case_names = list(train_pt["case_names"])
test_case_names = list(test_pt["case_names"])
n_train_cases = len(train_case_names)
all_case_names = train_case_names + test_case_names

# Offset test's patch_to_case indices by n_train_cases
train_p2c = train_pt["patch_to_case"]
test_p2c = test_pt["patch_to_case"] + n_train_cases
patch_to_case = torch.cat([train_p2c, test_p2c], dim=0)

# Merge wind_angles
wind_angles = list(train_pt["wind_angles"]) + list(test_pt["wind_angles"])

print(f"\nCombined: X={X.shape} ({X.shape[0]} patches, {len(all_case_names)} unique cases)")
assert X.shape[0] == len(patch_to_case), f"X rows {X.shape[0]} != patches {len(patch_to_case)}"

# Verify no NaN in X
assert not torch.isnan(X).any(), "X contains NaN!"

# Build bundle (v3 all_data.pt format)
bundle = {
    "X": X,
    "Y": Y,
    "nan_mask": nan_mask,
    "case_names": all_case_names,       # 512 unique case names
    "wind_angles": wind_angles,
    "patch_to_case": patch_to_case,      # 878 entries, indices into all_case_names
    "fmt_shape": train_pt["fmt_shape"],
}

# Backup old file and save new
out_path = V3_DIR / "all_data.pt"
if out_path.exists():
    backup = V3_DIR / "all_data.pt.v3_original.bak"
    print(f"Backing up old all_data.pt -> {backup.name}")
    import shutil
    shutil.copy2(out_path, backup)

torch.save(bundle, out_path)
print(f"Saved {out_path}")
print(f"  X={bundle['X'].shape} ({bundle['X'].shape[0]} patches)")
print(f"  Y={bundle['Y'].shape}")
print(f"  {len(all_case_names)} unique cases")
print(f"  fmt_shape={bundle['fmt_shape']}")

# Quick sanity: verify patch_to_case covers all cases
used_cases = set()
for i in range(len(patch_to_case)):
    used_cases.add(all_case_names[int(patch_to_case[i])])
print(f"  Cases referenced by patches: {len(used_cases)}")
assert len(used_cases) == len(all_case_names), f"Missing cases! {len(all_case_names) - len(used_cases)}"

# Verify split_manifest compatibility
import json
manifest_path = V3_DIR / "split_manifest.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    seed1 = manifest["seeds"]["1"]
    manifest_cases = set(seed1["train"]) | set(seed1["val"]) | set(seed1["holdout"])
    data_cases = set(all_case_names)
    overlap = manifest_cases & data_cases
    missing_from_data = manifest_cases - data_cases
    missing_from_manifest = data_cases - manifest_cases
    print(f"\n  Manifest seed=1: train={len(seed1['train'])} val={len(seed1['val'])} hold={len(seed1['holdout'])}")
    print(f"  Overlap with data: {len(overlap)}")
    if missing_from_data:
        print(f"  WARNING: {len(missing_from_data)} manifest cases not in data: {sorted(missing_from_data)[:5]}")
    if missing_from_manifest:
        print(f"  INFO: {len(missing_from_manifest)} data cases not in manifest (ok, extras ignored)")

# --- Load raw_cases from old all_data.pt or from raw files ---
old_all = V3_DIR / "all_data.pt.v3_original.bak"
raw_cases = None

if old_all.exists():
    print(f"\nLoading raw_cases from backup...")
    old_bundle = torch.load(old_all, map_location="cpu", weights_only=False)
    raw_cases = old_bundle.get("raw_cases", None)
    if raw_cases:
        # Filter to only 512 cases we have
        filtered = {k: v for k, v in raw_cases.items() if k in data_cases}
        print(f"  raw_cases from backup: {len(raw_cases)} total, {len(filtered)} in our 512 cases")
        raw_cases = filtered

if raw_cases is None:
    print(f"\nLoading raw_cases from raw files...")
    raw_cases = {}
    for cn in all_case_names:
        d = RAW_DIR / cn
        if not d.is_dir():
            continue
        m = re.search(r"_d(\d+)$", cn)
        angle = int(m.group(1)) if m else 0
        topo = np.flipud(np.loadtxt(d / f"{cn}_topo", dtype=np.float32))
        with xr.open_dataset(d / f"{cn}_ped.nc") as ds:
            uped = ds["Uped"].values.astype(np.float32)
        combined = uped.copy()
        combined[topo > 0] = -topo[topo > 0]
        raw_cases[cn] = (combined, angle)
    print(f"  Loaded raw_cases for {len(raw_cases)} cases")

bundle["raw_cases"] = raw_cases

# Re-save with raw_cases
torch.save(bundle, out_path)
print(f"Re-saved {out_path} with raw_cases ({len(raw_cases)} entries)")

print("\nDone!")
