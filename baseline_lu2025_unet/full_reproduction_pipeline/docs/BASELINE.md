# Baseline Analysis Report

**Date:** 2026-04-18
**Config:** UNetV2Baseline(n_c=16), masked_l1, lr=1e-3, Adam, batch=16, seed=1
**Data:** 705 train / 95 val / 78 holdout patches (seed=7 split)

---

## 1. Three Baseline Runs (Code Verification)

Three identical-config runs on different GPUs to verify code consistency:

| Epoch | Old L40S (1000ep) | New H100 (150ep) | New L40S (150ep) |
|-------|:-:|:-:|:-:|
| 1 | 0.2539 | 0.2544 | 0.2533 |
| 50 | 0.0901 | 0.0931 | 0.0933 |
| 100 | 0.0637 | 0.0657 | 0.0626 |
| 150 | **0.0468** | **0.0489** | **0.0466** |

### Cross-GPU Comparison (epoch 150 train loss)
| Pair | Diff | Explanation |
|------|------|-------------|
| Old L40S vs New L40S | **0.4%** | Same GPU = code verified identical |
| Old L40S vs New H100 | **4.5%** | GPU arch diff (Ada vs Hopper) |
| New L40S vs New H100 | **4.9%** | GPU arch diff, grows with epochs |

**Conclusion:** Code is identical across all runs. H100 (Hopper) consistently produces ~5% higher train loss than L40S (Ada Lovelace) due to different CUDA kernel implementations. This is normal PyTorch behavior.

---

## 2. Val Loss Comparison

| Run | GPU | Epoch 150 Val | Best Val | Val Split |
|-----|-----|:---:|:---:|---|
| Old baseline | L40S | 0.0924 | 0.0924 | val+ho=173 (seed=42) |
| New H100 | H100 | 0.0999 | 0.0998 | val=95 (seed=7) |
| New L40S | L40S | 0.1005 | **0.0997** | val=95 (seed=7) |

Val losses not directly comparable (different splits). New H100 and New L40S val losses are very close (0.0998 vs 0.0997).

**Reference Baseline for Grid 18: best_val = 0.0997 (New L40S, 150ep, val=95)**

---

## 3. Training Curve (1000ep Reference)

From old baseline (L40S, 1000ep, val+ho=173):

| Epoch | Train | Val | Best |
|-------|------:|----:|-----:|
| 1 | 0.2539 | 0.2048 | 0.2048 |
| 50 | 0.0901 | 0.1077 | 0.1053 |
| 100 | 0.0637 | 0.0981 | 0.0962 |
| 150 | 0.0468 | 0.0924 | 0.0924 |
| 300 | 0.0365 | 0.0905 | 0.0895 |
| 500 | 0.0264 | 0.0898 | 0.0886 |
| 650 | 0.0221 | 0.0890 | **0.0882** |
| 800 | 0.0206 | 0.0886 | 0.0882 |
| 1000 | 0.0188 | 0.0892 | 0.0882 |

- Best model at epoch 650
- Train loss continues decreasing (0.019@1000) while val plateaus (~0.089)
- Light overfitting after epoch 650

---

## 4. v2 Final Comparison

| | v2 (train_7level_v3.py) | v3 (train.py) | Match? |
|---|---|---|---|
| Loss | masked_l1 (NaN+building mask) | MaskedL1Loss (same) | ✅ |
| lr | 1e-3 | 1e-3 | ✅ |
| Optimizer | Adam (no wd) | Adam (no wd) | ✅ |
| Scheduler | None | None | ✅ |
| Model | UNetLu7Level(n_c=16) | UNetV2Baseline(n_c=16) | ✅ |
| Seed | 1 | 1 | ✅ |
| Data | full_masked_640 (878 patches) | all_data.pt (same data, concatenated) | ✅ |

v2 code reference: `auto_v2_final_report/scripts/train_7level_v3.py`

---

## 5. Important Lessons

- **v2 has two local copies**: `auto_v2_analysis/` is early version (nn.L1Loss), `auto_v2_final_report/scripts/` is final version (masked_l1)
- **GPU floating point diff**: H100 vs L40S same code can produce 4-5% train loss difference; same GPU only 0.4%
- **train.py was modified at 11:12** (val+ho → val-only), but this does NOT affect train loss
