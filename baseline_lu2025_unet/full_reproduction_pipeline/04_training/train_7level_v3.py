"""
Train 7-level UNet for a given seed, using Lu's exact train/test split.

Key: X_all has 878 patches, but only 512 unique cases.
Each case can have 1-9 patches (via DataFormatter's rotation/padding).
We use patch_to_case to correctly map ALL 878 patches to cases,
then split by Lu's per-seed case assignment.

Usage: python train_7level_v3.py --seed N
"""
import os, sys, argparse, logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.unet_lu_7level import UNetLu7Level

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

SEED = args.seed
EPOCHS = 1000
BATCH_SIZE = 16
LR = 1e-3
CKPT_INTERVAL = 100

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FMT_DIR = os.path.join(ROOT, "data", "full_masked_640")
RESULTS_DIR = os.path.join(ROOT, "results", "full_masked_640_7level", f"seed_{SEED}")
os.makedirs(RESULTS_DIR, exist_ok=True)
CKPT_PATH = os.path.join(RESULTS_DIR, "checkpoint.pt")

# =====================================================================
# Step 1: Load ALL formatted patches (train.pt + test.pt = 878 patches)
# =====================================================================
train_pt = torch.load(os.path.join(FMT_DIR, "train.pt"), map_location="cpu", weights_only=False)
test_pt = torch.load(os.path.join(FMT_DIR, "test.pt"), map_location="cpu", weights_only=False)

X_all = torch.cat([train_pt["X"], test_pt["X"]], dim=0)   # (878, 1, 640, 640)
Y_all = torch.cat([train_pt["Y"], test_pt["Y"]], dim=0)   # (878, 1, 640, 640)

# =====================================================================
# Step 2: Build per-patch -> case_name mapping using patch_to_case
# train_pt: 705 patches (case_names: 410 unique)
# test_pt:  173 patches (case_names: 102 unique)
# Total = 512 unique cases across 878 patches
# =====================================================================
train_cn = train_pt["case_names"]   # list of 410 case names
test_cn = test_pt["case_names"]     # list of 102 case names
train_p2c = train_pt["patch_to_case"]  # (705,) values 0..409
test_p2c = test_pt["patch_to_case"]    # (173,) values 0..101

patch_case_names = []  # length 878
for i in range(len(train_p2c)):
    patch_case_names.append(train_cn[int(train_p2c[i])])
for i in range(len(test_p2c)):
    patch_case_names.append(test_cn[int(test_p2c[i])])

assert len(patch_case_names) == X_all.shape[0], \
    f"patch_case_names ({len(patch_case_names)}) != X_all ({X_all.shape[0]})"

# Verify X has no NaN (NaN only in Y)
assert not torch.isnan(X_all).any(), "X_all contains NaN! This will corrupt training."

# case_name -> [patch_indices] (preserves encounter order = DataFormatter order)
case_to_patches = {}
for idx, cn in enumerate(patch_case_names):
    if cn not in case_to_patches:
        case_to_patches[cn] = []
    case_to_patches[cn].append(idx)

n_unique = len(case_to_patches)
n_patches_total = X_all.shape[0]
logging.info(f"Loaded {n_patches_total} patches, {n_unique} unique cases")

del train_pt, test_pt

# =====================================================================
# Step 3: Read Lu's train/test split for this seed
# =====================================================================
lu_train_csv = os.path.join(ROOT, "references", "all_cases_20exp",
                            f"metrics_in_training_set_seed{SEED}.csv")
lu_test_csv = os.path.join(ROOT, "references", "all_cases_20exp",
                           f"metrics_in_test_set_seed{SEED}.csv")

lu_train = pd.read_csv(lu_train_csv)
lu_test = pd.read_csv(lu_test_csv)

lu_train = lu_train[lu_train['topo'] != 'total']
lu_test = lu_test[lu_test['topo'] != 'total']

train_cases_lu = set(lu_train['topo'].astype(str) + '_d' +
                     lu_train['angle'].astype(str).str.zfill(2))
test_cases_lu = set(lu_test['topo'].astype(str) + '_d' +
                    lu_test['angle'].astype(str).str.zfill(2))

logging.info(f"Seed {SEED}: Lu train={len(train_cases_lu)} cases, "
             f"test={len(test_cases_lu)} cases")
assert len(train_cases_lu & test_cases_lu) == 0, "Train/test case overlap!"

# =====================================================================
# Step 4: Assign patches to train/test based on case membership
# ASSERT: every case in our data must appear in Lu's split
# =====================================================================
all_lu_cases = train_cases_lu | test_cases_lu
our_cases = set(case_to_patches.keys())
unmatched_ours = our_cases - all_lu_cases
unmatched_lu = all_lu_cases - our_cases
if unmatched_ours:
    logging.warning(f"Our cases not in Lu ({len(unmatched_ours)}): "
                    f"{sorted(unmatched_ours)[:5]}")
if unmatched_lu:
    logging.warning(f"Lu cases not in our data ({len(unmatched_lu)}): "
                    f"{sorted(unmatched_lu)[:5]}")

train_patch_indices = []
test_patch_indices = []
skipped_cases = []

for cn, patches in case_to_patches.items():
    if cn in test_cases_lu:
        test_patch_indices.extend(patches)
    elif cn in train_cases_lu:
        train_patch_indices.extend(patches)
    else:
        skipped_cases.append(cn)

train_patch_indices.sort()
test_patch_indices.sort()

logging.info(f"Train: {len(train_patch_indices)} patches from "
             f"{sum(1 for c in case_to_patches if c in train_cases_lu)} cases")
logging.info(f"Test:  {len(test_patch_indices)} patches from "
             f"{sum(1 for c in case_to_patches if c in test_cases_lu)} cases")

# CRITICAL ASSERTION: all patches must be assigned
total_assigned = len(train_patch_indices) + len(test_patch_indices)
assert total_assigned == n_patches_total, \
    f"Patch coverage: {total_assigned}/{n_patches_total} assigned. " \
    f"Skipped cases: {skipped_cases}"
if skipped_cases:
    logging.warning(f"Skipped {len(skipped_cases)} cases not in Lu's split")

X_train = X_all[train_patch_indices].clone()
Y_train = Y_all[train_patch_indices].clone()
X_test = X_all[test_patch_indices].clone()
Y_test = Y_all[test_patch_indices].clone()
del X_all, Y_all

# =====================================================================
# Step 5: Training setup
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
logging.info(f"Device: {device}, GPU: {gpu_name}")

X_train, Y_train = X_train.to(device), Y_train.to(device)
X_test, Y_test = X_test.to(device), Y_test.to(device)


def masked_l1(pred, target, mask):
    """Masked L1 loss: ignore NaN and building pixels (X > 0)."""
    return torch.where(mask, (pred - target).abs(),
                       torch.zeros_like(pred)).sum() / mask.sum().clamp(min=1)


model = UNetLu7Level(n_c=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
logging.info(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# ---- Resume from checkpoint if exists ----
start_epoch = 1
best_test_loss = float('inf')

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_test_loss = ckpt['best_test_loss']
    logging.info(f"Resumed from epoch {ckpt['epoch']}, "
                 f"best_test_loss={best_test_loss:.6f}")

# Verify batch size (FIX: clear gradients after probe to avoid leak)
try:
    p = model(X_train[:BATCH_SIZE])
    bm = (~torch.isnan(Y_train[:BATCH_SIZE])) & (X_train[:BATCH_SIZE] <= 0)
    loss = masked_l1(p, Y_train[:BATCH_SIZE], bm)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)  # FIX: clear probe gradients
    del p, loss
    torch.cuda.empty_cache()
    logging.info(f"Batch {BATCH_SIZE} OK")
except RuntimeError:
    torch.cuda.empty_cache()
    BATCH_SIZE = 8
    logging.warning(f"OOM with 16, using batch={BATCH_SIZE}")

train_loader = DataLoader(TensorDataset(X_train, Y_train),
                          batch_size=BATCH_SIZE, shuffle=True)

EVAL_BS = 16


def batch_eval(model, X, Y, bs):
    model.eval()
    with torch.no_grad():
        preds = [model(X[i:i+bs]) for i in range(0, len(X), bs)]
        pred = torch.cat(preds, dim=0)
        valid = (~torch.isnan(Y)) & (X <= 0)
        return masked_l1(pred, Y, valid).item()


# =====================================================================
# Step 6: Training loop
# =====================================================================
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    tot = 0.0
    n = 0
    for xb, yb in train_loader:
        pred = model(xb)
        bm = (~torch.isnan(yb)) & (xb <= 0)
        loss = masked_l1(pred, yb, bm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot += loss.item()
        n += 1
    tl = tot / n

    el = batch_eval(model, X_test, Y_test, EVAL_BS)

    is_new_best = el < best_test_loss
    if is_new_best:
        best_test_loss = el
        torch.save(model.state_dict(),
                   os.path.join(RESULTS_DIR, "model_best.pt"))

    if epoch % 10 == 0 or epoch == 1:
        mk = " *" if is_new_best else ""
        logging.info(f"Epoch {epoch:4d}/{EPOCHS}  train={tl:.6f}  "
                     f"test={el:.6f}  best={best_test_loss:.6f}{mk}")

    # Checkpoint every CKPT_INTERVAL epochs
    if epoch % CKPT_INTERVAL == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_test_loss': best_test_loss,
            'seed': SEED,
        }, CKPT_PATH)
        logging.info(f"Checkpoint saved at epoch {epoch}")

torch.save(model.state_dict(),
           os.path.join(RESULTS_DIR, "model_final.pt"))
if os.path.exists(CKPT_PATH):
    os.remove(CKPT_PATH)
logging.info(f"Seed {SEED} done. Best test loss: {best_test_loss:.6f}")
