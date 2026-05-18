# Hybrid Initial Knowledge Package

## 0. Provenance

- Source session / project: Grid, controller_grid_controller_001 to controller_grid_controller_020
- Authoring agent: Hermes Agent in the human researcher's Grid workspace
- Date: 2026-05-06 EDT
- Data source: local Grid artifacts under `<LOCAL_HOME_PATH>`
- Any known deviations from Hybrid contract: This package contains only grid_controller / Grid controller evidence. It does not import any other phase result, ranking, proposal rationale, or report.
- Holdout exclusion rule: this package must contain **no holdout metrics, no holdout rankings, no holdout-derived conclusions, and no candidate selection based on holdout**. Hybrid initial knowledge may use only training/validation-side evidence from the supplied grid_controller / Grid package. Holdout is reserved for selected/final evaluation only and must not guide planner search.

## 1. Executive summary

- grid_controller covered 20 controller rounds and 200 source candidates from `grid_curated_002`.
- The full ledger resolves to 172 benchmark200 ranked rows and 28 formal parameter-cap skipped rows, with no missing or inconsistent rows in the generated ledger.
- The search baseline is `orthogonal exploratory sweep_200ep_baseline` from `baseline_source/campaigns/orthogonal exploratory sweep_200ep/baseline`, with val R2_median 0.708503411.
- Only 8 of 172 benchmark200 ranked rows exceeded the val R2_median baseline, so the median-R2 gate is strict.
- `unet_sdf_7level` is the strongest and most stable family: 5/5 ranked rows exceeded the val R2_median baseline.
- Oversized mamba2d, ufno, hrformer, and unet_afno variants should not be repeated unless redesigned below the 150M formal cap.

## 2. Baseline and metric definitions

- Search metric: primary search metric is val R2_median. Secondary evidence includes val R2_global and val MAE_median.
- Validation split size: grid_controller selected candidate val per-case metrics contain 55 val cases.
- Baseline model: `orthogonal exploratory sweep_200ep_baseline`, architecture `unet_v2_baseline`, source `<BASELINE_HPC_SOURCE_ROOT>/campaigns/orthogonal exploratory sweep_200ep/baseline`.
- Baseline metric values:

| split | source | R2_median | R2_mean | R2_global | MAE_median | MAE_mean | n_cases | use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | grid/shared/data/grid_baseline_reference.json | 0.708503411 | 0.622875278 | 0.656619483 | 0.093510993 | 0.085752743 | 55 for grid_controller candidate val metrics; baseline reference stores scalar values | search and benchmark200 screening |


## 3. Full model result table

The full one-row-per-source-candidate table is written to the companion CSV:

```text
<LOCAL_HOME_PATH>
```

The CSV has 200 rows and uses the exact requested columns:

```text
```

For readability, the highest 30 benchmark200 rows by val R2_median are excerpted here.

| --- | --- | --- | --- | --- | --- | --- | --- |
| grid_controller_003_05_unet_sdf_7level | unet_sdf_7level | 125,771,905 (125.8M) | 0.734949 | 0.702937 | 0.085048 | 0.749991 | benchmark200_ranked |
| grid_controller_017_06_unet_sdf_7level | unet_sdf_7level | 125,771,905 (125.8M) | 0.729361 | 0.709460 | 0.086881 | 0.766345 | benchmark200_ranked |
| grid_controller_018_04_unet_sdf_7level | unet_sdf_7level | 125,771,905 (125.8M) | 0.726549 | 0.710891 | 0.084685 | 0.774666 | benchmark200_ranked |
| grid_controller_011_03_fourier_unet | fourier_unet | 34,201,393 (34.2M) | 0.721341 | 0.690672 | 0.083326 | 0.749812 | benchmark200_ranked |
| grid_controller_019_04_unet_sdf_7level | unet_sdf_7level | 125,771,905 (125.8M) | 0.721115 | 0.700792 | 0.084222 | 0.761668 | benchmark200_ranked |
| grid_controller_009_04_fourier_unet | fourier_unet | 34,201,393 (34.2M) | 0.710953 | 0.686029 | 0.087191 | 0.752305 | benchmark200_ranked |
| grid_controller_019_00_unet_sdf_7level | unet_sdf_7level | 125,771,905 (125.8M) | 0.708871 | 0.692494 | 0.088401 | 0.725684 | benchmark200_ranked |
| grid_controller_019_05_unet_v2_baseline | unet_v2_baseline | 34,610,193 (34.6M) | 0.708585 | 0.691738 | 0.087149 | 0.714447 | benchmark200_ranked |
| grid_controller_004_05_cno_v2 | cno_v2 | 144,933,361 (144.9M) | 0.705373 | 0.684045 | 0.086512 |  | benchmark200_ranked |
| grid_controller_007_09_dilated_unet | dilated_unet | 14,780,705 (14.8M) | 0.703875 | 0.669857 | 0.088735 |  | benchmark200_ranked |
| grid_controller_012_06_multiscale_conv | multiscale_conv | 75,159,889 (75.2M) | 0.702762 | 0.687898 | 0.085647 |  | benchmark200_ranked |
| grid_controller_006_05_cno_v2 | cno_v2 | 144,933,361 (144.9M) | 0.700826 | 0.674971 | 0.090017 |  | benchmark200_ranked |
| grid_controller_004_02_attention_mamba | attention_mamba | 35,969,713 (36.0M) | 0.696389 | 0.671235 | 0.086492 |  | benchmark200_ranked |
| grid_controller_002_06_quadmamba | quadmamba | 132,194,421 (132.2M) | 0.696235 | 0.675200 | 0.086326 |  | benchmark200_ranked |
| grid_controller_020_09_unet_v2_baseline | unet_v2_baseline | 34,610,193 (34.6M) | 0.695249 | 0.683756 | 0.093230 | 0.723541 | benchmark200_ranked |
| grid_controller_005_02_attention_mamba | attention_mamba | 35,969,713 (36.0M) | 0.692225 | 0.660658 | 0.088225 |  | benchmark200_ranked |
| grid_controller_014_01_multiscale_conv | multiscale_conv | 75,159,889 (75.2M) | 0.691417 | 0.682706 | 0.087844 |  | benchmark200_ranked |
| grid_controller_004_08_dilated_fno | dilated_fno | 60,726,001 (60.7M) | 0.691132 | 0.665984 | 0.087023 |  | benchmark200_ranked |
| grid_controller_019_02_unet_v3 | unet_v3 | 138,432,529 (138.4M) | 0.690247 | 0.675043 | 0.089471 |  | benchmark200_ranked |
| grid_controller_015_05_sac_unet | sac_unet | 133,726,297 (133.7M) | 0.690088 | 0.670633 | 0.091746 |  | benchmark200_ranked |
| grid_controller_004_01_attention_gate_unet | attention_gate_unet | 31,467,765 (31.5M) | 0.688523 | 0.661787 | 0.092307 |  | benchmark200_ranked |
| grid_controller_012_01_fourier_unet | fourier_unet | 34,201,393 (34.2M) | 0.686502 | 0.653833 | 0.087260 |  | benchmark200_ranked |
| grid_controller_001_01_attention_mamba | attention_mamba | 35,969,713 (36.0M) | 0.686384 | 0.651668 | 0.092644 |  | benchmark200_ranked |
| grid_controller_005_05_cno_v2 | cno_v2 | 144,933,361 (144.9M) | 0.685325 | 0.673261 | 0.089400 |  | benchmark200_ranked |
| grid_controller_018_06_unet_v3 | unet_v3 | 138,432,529 (138.4M) | 0.685211 | 0.673424 | 0.092541 |  | benchmark200_ranked |
| grid_controller_008_03_dilated_unet | dilated_unet | 14,780,705 (14.8M) | 0.684894 | 0.679513 | 0.091503 |  | benchmark200_ranked |
| grid_controller_010_04_fourier_unet | fourier_unet | 34,201,393 (34.2M) | 0.682787 | 0.642210 | 0.088299 |  | benchmark200_ranked |
| grid_controller_003_09_attention_gate_unet | attention_gate_unet | 31,467,765 (31.5M) | 0.681540 | 0.667969 | 0.092386 |  | benchmark200_ranked |
| grid_controller_003_06_unet_v3 | unet_v3 | 138,432,529 (138.4M) | 0.680521 | 0.690057 | 0.089692 |  | benchmark200_ranked |
| grid_controller_010_00_dilated_unet | dilated_unet | 14,780,705 (14.8M) | 0.678897 | 0.691298 | 0.089102 |  | benchmark200_ranked |

## 4. Per-family synthesis

### Architecture-level summary table

| arch | source | ranked | skipped | params | best_val_R2med | median_val_R2med | n>val_base | n>global_base | n_better_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unet_sdf_7level | 5 | 5 | 0 | 125,771,905 (125.8M) | 0.734949 | 0.726549 | 5 | 5 | 5 |
| fourier_unet | 5 | 5 | 0 | 34,201,393 (34.2M) | 0.721341 | 0.686502 | 2 | 3 | 5 |
| unet_v2_baseline | 16 | 16 | 0 | 34,610,193 (34.6M) | 0.708585 | 0.656163 | 1 | 16 | 6 |
| cno_v2 | 5 | 5 | 0 | 144,933,361 (144.9M) | 0.705373 | 0.685325 | 0 | 3 | 4 |
| dilated_unet | 5 | 5 | 0 | 14,780,705 (14.8M) | 0.703875 | 0.678897 | 0 | 4 | 4 |
| multiscale_conv | 5 | 5 | 0 | 75,159,889 (75.2M) | 0.702762 | 0.678585 | 0 | 5 | 5 |
| attention_mamba | 5 | 5 | 0 | 35,969,713 (36.0M) | 0.696389 | 0.686384 | 0 | 3 | 4 |
| quadmamba | 5 | 1 | 4 | 132.2M-338.4M | 0.696235 | 0.696235 | 0 | 1 | 1 |
| dilated_fno | 5 | 5 | 0 | 60,726,001 (60.7M) | 0.691132 | 0.659440 | 0 | 1 | 4 |
| unet_v3 | 5 | 5 | 0 | 138,432,529 (138.4M) | 0.690247 | 0.680521 | 0 | 5 | 4 |
| sac_unet | 5 | 5 | 0 | 133,726,297 (133.7M) | 0.690088 | 0.669487 | 0 | 1 | 3 |
| attention_gate_unet | 5 | 5 | 0 | 31,467,765 (31.5M) | 0.688523 | 0.669671 | 0 | 4 | 2 |
| umamba | 5 | 5 | 0 | 72,089,137 (72.1M) | 0.674408 | 0.664584 | 0 | 3 | 0 |
| dilated_hrformer | 5 | 5 | 0 | 42,474,673 (42.5M) | 0.672146 | 0.550113 | 0 | 1 | 0 |
| hrdcn | 5 | 5 | 0 | 9,258,281 (9.3M) | 0.663423 | 0.647321 | 0 | 0 | 0 |
| hrnet | 5 | 5 | 0 | 20,138,689 (20.1M) | 0.661625 | 0.651748 | 0 | 2 | 0 |
| cbam_unet | 5 | 5 | 0 | 31,340,033 (31.3M) | 0.657932 | -3.323431 | 0 | 0 | 0 |
| dcn_unet | 5 | 5 | 0 | 135,192,475 (135.2M) | 0.632272 | 0.627985 | 0 | 0 | 0 |
| residual_spectral | 5 | 1 | 4 | 105.0M-171.0M | 0.628984 | 0.628984 | 0 | 0 | 0 |
| sac_mamba | 5 | 5 | 0 | 16,552,777 (16.6M) | 0.623816 | 0.580979 | 0 | 0 | 0 |
| mamba_attention | 5 | 5 | 0 | 22,135,133 (22.1M) | 0.618517 | 0.585293 | 0 | 0 | 0 |
| fno_encoder_decoder | 5 | 5 | 0 | 62,924,353 (62.9M) | 0.606920 | 0.548157 | 0 | 0 | 0 |
| swin_unetr | 5 | 5 | 0 | 6,859,867 (6.9M) | 0.539313 | 0.526485 | 0 | 0 | 0 |
| ffno | 5 | 5 | 0 | 206,337 (0.2M) | 0.470370 | 0.450553 | 0 | 0 | 0 |
| transolver | 5 | 5 | 0 | 121,316,065 (121.3M) | 0.413157 | 0.401936 | 0 | 0 | 0 |
| kan_unet | 5 | 5 | 0 | 21,756,196 (21.8M) | 0.393951 | 0.343497 | 0 | 0 | 0 |
| convnext_v2_unet | 5 | 5 | 0 | 7,463,841 (7.5M) | 0.387231 | 0.348415 | 0 | 0 | 0 |
| fno2d | 5 | 5 | 0 | 42.8M-96.4M | 0.346560 | 0.337540 | 0 | 0 | 0 |
| uno | 5 | 5 | 0 | 105,601,345 (105.6M) | 0.342392 | 0.327142 | 0 | 0 | 0 |
| afno | 5 | 5 | 0 | 0.9M-11.9M | 0.243193 | 0.185428 | 0 | 0 | 0 |
| nafnet | 4 | 4 | 0 | 2,859,313 (2.9M) | 0.009407 | 0.008129 | 0 | 0 | 0 |
| perceiver_io | 5 | 5 | 0 | 7,082,753 (7.1M) | -0.017238 | -3.323431 | 0 | 0 | 0 |
| transolver_lite | 5 | 5 | 0 | 1,023,425 (1.0M) | -1.238094 | -1.407186 | 0 | 0 | 0 |
| cnn_deeponet | 5 | 5 | 0 | 1,680,577 (1.7M) | -3.323431 | -3.323431 | 0 | 0 | 0 |
| hrformer | 5 | 0 | 5 | 215,350,593 (215.4M) |  |  | 0 | 0 | 0 |
| mamba2d | 5 | 0 | 5 | 423,114,785 (423.1M) |  |  | 0 | 0 | 0 |
| ufno | 5 | 0 | 5 | 242,225,153 (242.2M) |  |  | 0 | 0 | 0 |
| unet_afno | 5 | 0 | 5 | 188,782,610 (188.8M) |  |  | 0 | 0 | 0 |

### unet_sdf_7level

- Best model_id: grid_controller_003_05_unet_sdf_7level
- Parameter efficiency: parameter range 125.8M; best row params 125,771,905 (125.8M).
- Failure modes: no repeated execution failure observed in the available evidence.
- What seems not to work: none evident, but it is relatively large at about 125.8M params.
- Confidence level: high
- Suggested Hybrid action: exploit

### fourier_unet

- Best model_id: grid_controller_011_03_fourier_unet
- Parameter efficiency: parameter range 34.2M; best row params 34,201,393 (34.2M).
- Failure modes: no repeated execution failure observed in the available evidence.
- What seems not to work: performance varied by source candidate, so seed/config sensitivity remains.
- Confidence level: high
- Suggested Hybrid action: exploit / ablate

### unet_v2_baseline

- Best model_id: grid_controller_019_05_unet_v2_baseline
- Parameter efficiency: parameter range 34.6M; best row params 34,610,193 (34.6M).
- Failure modes: no repeated execution failure observed in the available evidence.
- What seems to work: stable control family with strong R2_global relative to baseline.
- What seems not to work: only one of sixteen rows barely exceeded val R2_median baseline.
- Confidence level: high
- Suggested Hybrid action: control

### cno_v2

- Best model_id: grid_controller_004_05_cno_v2
- Main result: best val R2_median 0.705373, median across ranked rows 0.685325, ranked 5/5.
- Parameter efficiency: parameter range 144.9M; best row params 144,933,361 (144.9M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: some benchmark200 rows approached the val baseline or exceeded global/MAE criteria.
- What seems not to work: did not consistently exceed val R2_median baseline.
- Confidence level: medium
- Suggested Hybrid action: ablate / control

### dilated_unet

- Best model_id: grid_controller_007_09_dilated_unet
- Main result: best val R2_median 0.703875, median across ranked rows 0.678897, ranked 5/5.
- Parameter efficiency: parameter range 14.8M; best row params 14,780,705 (14.8M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: some benchmark200 rows approached the val baseline or exceeded global/MAE criteria.
- What seems not to work: did not consistently exceed val R2_median baseline.
- Confidence level: medium
- Suggested Hybrid action: ablate / control

### multiscale_conv

- Best model_id: grid_controller_012_06_multiscale_conv
- Main result: best val R2_median 0.702762, median across ranked rows 0.678585, ranked 5/5.
- Parameter efficiency: parameter range 75.2M; best row params 75,159,889 (75.2M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: some benchmark200 rows approached the val baseline or exceeded global/MAE criteria.
- What seems not to work: did not consistently exceed val R2_median baseline.
- Confidence level: medium
- Suggested Hybrid action: ablate / control

### attention_mamba

- Best model_id: grid_controller_004_02_attention_mamba
- Main result: best val R2_median 0.696389, median across ranked rows 0.686384, ranked 5/5.
- Parameter efficiency: parameter range 36.0M; best row params 35,969,713 (36.0M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: some benchmark200 rows approached the val baseline or exceeded global/MAE criteria.
- What seems not to work: did not consistently exceed val R2_median baseline.
- Confidence level: medium
- Suggested Hybrid action: ablate / control

### quadmamba

- Best model_id: grid_controller_002_06_quadmamba
- Main result: best val R2_median 0.696235, median across ranked rows 0.696235, ranked 1/5.
- Parameter efficiency: parameter range 132.2M-338.4M; best row params 132,194,421 (132.2M).
- Failure modes: 4 source candidates exceeded the 150M formal benchmark cap or were skipped after smoke evidence.
- What seems to work: mechanism may be feasible only after down-scaling.
- What seems not to work: current generated variants are too large for the formal 150M cap.
- Confidence level: high
- Suggested Hybrid action: avoid unless scaled down

### dilated_fno

- Best model_id: grid_controller_004_08_dilated_fno
- Main result: best val R2_median 0.691132, median across ranked rows 0.659440, ranked 5/5.
- Parameter efficiency: parameter range 60.7M; best row params 60,726,001 (60.7M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: some benchmark200 rows approached the val baseline or exceeded global/MAE criteria.
- What seems not to work: did not consistently exceed val R2_median baseline.
- Confidence level: medium
- Suggested Hybrid action: ablate / control

### unet_v3

- Best model_id: grid_controller_019_02_unet_v3
- Parameter efficiency: parameter range 138.4M; best row params 138,432,529 (138.4M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems not to work: val R2_median alone under-ranked it in this evidence set.
- Confidence level: medium
- Suggested Hybrid action: explore further / control

### sac_unet

- Best model_id: grid_controller_015_05_sac_unet
- Main result: best val R2_median 0.690088, median across ranked rows 0.669487, ranked 5/5.
- Parameter efficiency: parameter range 133.7M; best row params 133,726,297 (133.7M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: some benchmark200 rows approached the val baseline or exceeded global/MAE criteria.
- What seems not to work: did not consistently exceed val R2_median baseline.
- Confidence level: medium
- Suggested Hybrid action: ablate / control

### attention_gate_unet

- Best model_id: grid_controller_004_01_attention_gate_unet
- Main result: best val R2_median 0.688523, median across ranked rows 0.669671, ranked 5/5.
- Parameter efficiency: parameter range 31.5M; best row params 31,467,765 (31.5M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: runs completed and may provide auxiliary controls.
- What seems not to work: no clear median-R2 advantage over baseline.
- Confidence level: medium
- Suggested Hybrid action: control / low-priority ablate

### umamba

- Best model_id: grid_controller_018_02_umamba
- Main result: best val R2_median 0.674408, median across ranked rows 0.664584, ranked 5/5.
- Parameter efficiency: parameter range 72.1M; best row params 72,089,137 (72.1M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: runs completed and may provide auxiliary controls.
- What seems not to work: no clear median-R2 advantage over baseline.
- Confidence level: medium
- Suggested Hybrid action: control / low-priority ablate

### dilated_hrformer

- Best model_id: grid_controller_005_09_dilated_hrformer
- Main result: best val R2_median 0.672146, median across ranked rows 0.550113, ranked 5/5.
- Parameter efficiency: parameter range 42.5M; best row params 42,474,673 (42.5M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: runs completed and may provide auxiliary controls.
- What seems not to work: no clear median-R2 advantage over baseline.
- Confidence level: medium
- Suggested Hybrid action: control / low-priority ablate

### hrdcn

- Best model_id: grid_controller_008_08_hrdcn
- Main result: best val R2_median 0.663423, median across ranked rows 0.647321, ranked 5/5.
- Parameter efficiency: parameter range 9.3M; best row params 9,258,281 (9.3M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: runs completed and may provide auxiliary controls.
- What seems not to work: no clear median-R2 advantage over baseline.
- Confidence level: medium
- Suggested Hybrid action: control / low-priority ablate

### hrnet

- Best model_id: grid_controller_009_07_hrnet
- Main result: best val R2_median 0.661625, median across ranked rows 0.651748, ranked 5/5.
- Parameter efficiency: parameter range 20.1M; best row params 20,138,689 (20.1M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: runs completed and may provide auxiliary controls.
- What seems not to work: no clear median-R2 advantage over baseline.
- Confidence level: medium
- Suggested Hybrid action: control / low-priority ablate

### cbam_unet

- Best model_id: grid_controller_006_03_cbam_unet
- Main result: best val R2_median 0.657932, median across ranked rows -3.323431, ranked 5/5.
- Parameter efficiency: parameter range 31.3M; best row params 31,340,033 (31.3M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: runs completed and may provide auxiliary controls.
- What seems not to work: no clear median-R2 advantage over baseline.
- Confidence level: medium
- Suggested Hybrid action: control / low-priority ablate

### dcn_unet

- Best model_id: grid_controller_007_06_dcn_unet
- Main result: best val R2_median 0.632272, median across ranked rows 0.627985, ranked 5/5.
- Parameter efficiency: parameter range 135.2M; best row params 135,192,475 (135.2M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: runs completed and may provide auxiliary controls.
- What seems not to work: no clear median-R2 advantage over baseline.
- Confidence level: medium
- Suggested Hybrid action: control / low-priority ablate

### residual_spectral

- Best model_id: grid_controller_002_07_residual_spectral
- Main result: best val R2_median 0.628984, median across ranked rows 0.628984, ranked 1/5.
- Parameter efficiency: parameter range 105.0M-171.0M; best row params 104,972,257 (105.0M).
- Failure modes: 4 source candidates exceeded the 150M formal benchmark cap or were skipped after smoke evidence.
- What seems to work: mechanism may be feasible only after down-scaling.
- What seems not to work: current generated variants are too large for the formal 150M cap.
- Confidence level: high
- Suggested Hybrid action: avoid unless scaled down

### sac_mamba

- Best model_id: grid_controller_016_02_sac_mamba
- Main result: best val R2_median 0.623816, median across ranked rows 0.580979, ranked 5/5.
- Parameter efficiency: parameter range 16.6M; best row params 16,552,777 (16.6M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: runs completed and may provide auxiliary controls.
- What seems not to work: no clear median-R2 advantage over baseline.
- Confidence level: medium
- Suggested Hybrid action: control / low-priority ablate

### mamba_attention

- Best model_id: grid_controller_013_01_mamba_attention
- Main result: best val R2_median 0.618517, median across ranked rows 0.585293, ranked 5/5.
- Parameter efficiency: parameter range 22.1M; best row params 22,135,133 (22.1M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: runs completed and may provide auxiliary controls.
- What seems not to work: no clear median-R2 advantage over baseline.
- Confidence level: medium
- Suggested Hybrid action: control / low-priority ablate

### fno_encoder_decoder

- Best model_id: grid_controller_010_03_fno_encoder_decoder
- Main result: best val R2_median 0.606920, median across ranked rows 0.548157, ranked 5/5.
- Parameter efficiency: parameter range 62.9M; best row params 62,924,353 (62.9M).
- Failure modes: benchmark200 completed, but no row exceeded val R2_median baseline.
- What seems to work: runs completed and may provide auxiliary controls.
- What seems not to work: no clear median-R2 advantage over baseline.
- Confidence level: medium
- Suggested Hybrid action: control / low-priority ablate

### swin_unetr

- Best model_id: grid_controller_002_09_swin_unetr
- Main result: best val R2_median 0.539313, median across ranked rows 0.526485, ranked 5/5.
- Parameter efficiency: parameter range 6.9M; best row params 6,859,867 (6.9M).
- Failure modes: benchmark200 completed but val R2_median remained far below baseline.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### ffno

- Best model_id: grid_controller_008_04_ffno
- Main result: best val R2_median 0.470370, median across ranked rows 0.450553, ranked 5/5.
- Parameter efficiency: parameter range 0.2M; best row params 206,337 (0.2M).
- Failure modes: benchmark200 completed but val R2_median remained far below baseline.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### transolver

- Best model_id: grid_controller_017_01_transolver
- Main result: best val R2_median 0.413157, median across ranked rows 0.401936, ranked 5/5.
- Parameter efficiency: parameter range 121.3M; best row params 121,316,065 (121.3M).
- Failure modes: benchmark200 completed but val R2_median remained far below baseline.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### kan_unet

- Best model_id: grid_controller_010_08_kan_unet
- Main result: best val R2_median 0.393951, median across ranked rows 0.343497, ranked 5/5.
- Parameter efficiency: parameter range 21.8M; best row params 21,756,196 (21.8M).
- Failure modes: benchmark200 completed but val R2_median remained far below baseline.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### convnext_v2_unet

- Best model_id: grid_controller_006_06_convnext_v2_unet
- Main result: best val R2_median 0.387231, median across ranked rows 0.348415, ranked 5/5.
- Parameter efficiency: parameter range 7.5M; best row params 7,463,841 (7.5M).
- Failure modes: benchmark200 completed but val R2_median remained far below baseline.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### fno2d

- Best model_id: grid_controller_008_05_fno2d
- Main result: best val R2_median 0.346560, median across ranked rows 0.337540, ranked 5/5.
- Parameter efficiency: parameter range 42.8M-96.4M; best row params 42,833,473 (42.8M).
- Failure modes: benchmark200 completed but val R2_median remained far below baseline.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### uno

- Best model_id: grid_controller_017_09_uno
- Main result: best val R2_median 0.342392, median across ranked rows 0.327142, ranked 5/5.
- Parameter efficiency: parameter range 105.6M; best row params 105,601,345 (105.6M).
- Failure modes: benchmark200 completed but val R2_median remained far below baseline.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### afno

- Best model_id: grid_controller_001_00_afno
- Main result: best val R2_median 0.243193, median across ranked rows 0.185428, ranked 5/5.
- Parameter efficiency: parameter range 0.9M-11.9M; best row params 11,925,985 (11.9M).
- Failure modes: benchmark200 completed but val R2_median remained far below baseline.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### nafnet

- Best model_id: grid_controller_012_07_nafnet
- Main result: best val R2_median 0.009407, median across ranked rows 0.008129, ranked 4/4.
- Parameter efficiency: parameter range 2.9M; best row params 2,859,313 (2.9M).
- Failure modes: benchmark200 completed but val R2_median remained far below baseline.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### perceiver_io

- Best model_id: grid_controller_013_04_perceiver_io
- Main result: best val R2_median -0.017238, median across ranked rows -3.323431, ranked 5/5.
- Parameter efficiency: parameter range 7.1M; best row params 7,082,753 (7.1M).
- Failure modes: benchmark200 completed but predictive performance was negative on val R2_median.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### transolver_lite

- Best model_id: grid_controller_015_08_transolver_lite
- Main result: best val R2_median -1.238094, median across ranked rows -1.407186, ranked 5/5.
- Parameter efficiency: parameter range 1.0M; best row params 1,023,425 (1.0M).
- Failure modes: benchmark200 completed but predictive performance was negative on val R2_median.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### cnn_deeponet

- Best model_id: grid_controller_001_02_cnn_deeponet
- Main result: best val R2_median -3.323431, median across ranked rows -3.323431, ranked 5/5.
- Parameter efficiency: parameter range 1.7M; best row params 1,680,577 (1.7M).
- Failure modes: benchmark200 completed but predictive performance was negative on val R2_median.
- What seems to work: no clear positive signal in current grid_controller evidence.
- What seems not to work: low val R2_median or unstable behavior relative to baseline.
- Confidence level: medium
- Suggested Hybrid action: avoid

### hrformer

- Best model_id: none
- Main result: no benchmark200 ranked rows, 5/5 skipped by formal parameter cap.
- Parameter efficiency: parameter range 215.4M; best row params 215.4M.
- Failure modes: 5 source candidates exceeded the 150M formal benchmark cap or were skipped after smoke evidence.
- What seems to work: mechanism may be feasible only after down-scaling.
- What seems not to work: current generated variants are too large for the formal 150M cap.
- Confidence level: high
- Suggested Hybrid action: avoid unless scaled down

### mamba2d

- Best model_id: none
- Main result: no benchmark200 ranked rows, 5/5 skipped by formal parameter cap.
- Parameter efficiency: parameter range 423.1M; best row params 423.1M.
- Failure modes: 5 source candidates exceeded the 150M formal benchmark cap or were skipped after smoke evidence.
- What seems to work: mechanism may be feasible only after down-scaling.
- What seems not to work: current generated variants are too large for the formal 150M cap.
- Confidence level: high
- Suggested Hybrid action: avoid unless scaled down

### ufno

- Best model_id: none
- Main result: no benchmark200 ranked rows, 5/5 skipped by formal parameter cap.
- Parameter efficiency: parameter range 242.2M; best row params 242.2M.
- Failure modes: 5 source candidates exceeded the 150M formal benchmark cap or were skipped after smoke evidence.
- What seems to work: mechanism may be feasible only after down-scaling.
- What seems not to work: current generated variants are too large for the formal 150M cap.
- Confidence level: high
- Suggested Hybrid action: avoid unless scaled down

### unet_afno

- Best model_id: none
- Main result: no benchmark200 ranked rows, 5/5 skipped by formal parameter cap.
- Parameter efficiency: parameter range 188.8M; best row params 188.8M.
- Failure modes: 5 source candidates exceeded the 150M formal benchmark cap or were skipped after smoke evidence.
- What seems to work: mechanism may be feasible only after down-scaling.
- What seems not to work: current generated variants are too large for the formal 150M cap.
- Confidence level: high
- Suggested Hybrid action: avoid unless scaled down


## 5. Hypotheses for Hybrid

### H-001: SDF-augmented deep UNet is the primary exploit path

- Claim: A 7-level UNet with SDF mechanism is the strongest Hybrid seed family among grid_controller evidence.
- Proposed Hybrid test: run 3 to 5 SDF-UNet variants below 150M params with matched training protocol.
- Paired comparison / control: `unet_v2_baseline`, `unet_v3`, and a same-depth no-SDF UNet when available.
- Expected failure interpretation: if same-depth no-SDF control matches it, the gain may be due to depth/scale rather than SDF.
- Risk / resource expectation: moderate-high params around 125.8M, still within the 150M formal cap.

### H-002: Fourier-UNet provides a parameter-efficient second exploit path

- Claim: Fourier-UNet is the strongest lower-parameter alternative to SDF-UNet.
- Proposed Hybrid test: run multiple Fourier-UNet seeds/configs with the same data split and 200ep screening protocol.
- Paired comparison / control: `unet_v2_baseline` at comparable parameter scale.
- Expected failure interpretation: high variance would indicate config sensitivity rather than family-level failure.
- Risk / resource expectation: moderate, about 34.2M params.

### H-003: Val R2_median alone can under-rank some useful candidates

- Claim: Hybrid should not rely only on val R2_median when a candidate has strong R2_global or MAE signals.
- Proposed Hybrid test: add a secondary selection gate using R2_global and MAE_median for a small number of candidates.
- Paired comparison / control: select one candidate by val R2_median only and one by combined val R2_global / MAE signal.
- Expected failure interpretation: `unet_v3` was an isolated split effect.

### H-004: Oversized operator families need explicit cap-aware redesign before reuse

- Claim: Current Mamba2D, UFNO, HRFormer, and Unet-AFNO variants should not be repeated in Hybrid without down-scaling.
- Evidence from source session: mamba2d, ufno, hrformer, and unet_afno had all five source candidates skipped by the 150M formal cap.
- Proposed Hybrid test: only generate compact variants with preflight parameter estimates below 150M.
- Paired comparison / control: no formal benchmark should be allocated to an oversized variant.
- Decision rule: reject before benchmark if estimated params exceed 150M.
- Expected success: fewer smoke-passed but benchmark-skipped candidates.
- Expected failure interpretation: if compact variants still underperform, mechanism is low priority for this data.
- Risk / resource expectation: high if not capped, moderate if preflight param checks are enforced.

### H-005: Baseline UNet remains a required control but not the main improvement route

- Claim: `unet_v2_baseline` should stay as an Hybrid control, not as the main exploit family.
- Proposed Hybrid test: include one baseline-compatible control per batch or phase.
- Paired comparison / control: compare every exploit candidate to `orthogonal exploratory sweep_200ep_baseline` and an in-campaign `unet_v2_baseline` control.
- Decision rule: exploit family must beat both the stored orthogonal exploratory sweep baseline and contemporaneous control.
- Expected success: stable calibration of search drift.
- Expected failure interpretation: if controls fluctuate strongly, split or training reproducibility should be audited.
- Risk / resource expectation: low, about 34.6M params.


## 6. Candidate seeds for Hybrid planner

These are planner guidance seeds, not mandatory configs.

| priority | role | mechanism | suggested config sketch | paired control | rationale | avoid conditions |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | control | UNet v2 baseline | orthogonal exploratory sweep-compatible unet_v2_baseline, n_c=16, 200ep | orthogonal exploratory sweep baseline source | stable reference family and direct baseline continuity | do not present as new performance claim |
| 5 | ablation | SDF contribution | pair unet_sdf_7level with no-SDF 7-level UNet at similar params | same depth/width without SDF | tests whether SDF mechanism or depth/scale produced the gain | avoid if paired control exceeds 150M |
| 6 | avoid | oversized Mamba/HRFormer/UFNO | only run scaled-down variants below 150M | none unless cap-compliant | many source candidates skipped after smoke due parameter cap | do not repeat current oversized configs |

Roles: exploit, ablation, control, explorer.

## 7. Negative knowledge / avoid list

| item | evidence | reason to avoid | exception condition |
| --- | --- | --- | --- |
| mamba2d current configs | 5 source candidates, 5 param-cap skipped, 423.1M params | exceeds 150M formal cap | only if redesigned below cap with clear memory plan |
| ufno current configs | 5 source candidates, 5 param-cap skipped, 242.2M params | exceeds 150M formal cap | only if scaled down and smoke memory evidence is clean |
| hrformer current configs | 5 source candidates, 5 param-cap skipped, 215.4M params | exceeds 150M formal cap | only if a compact HRFormer is generated |
| unet_afno current configs | 5 source candidates, 5 param-cap skipped, 188.8M params | exceeds formal cap | only if AFNO bottleneck is reduced below cap |
| transolver_lite current configs | 5 ranked rows, best val R2_median -1.238094 | poor predictive performance despite small size | only as a diagnostic failure control |
| cnn_deeponet current configs | 5 ranked rows, val R2_median -3.323431 for all rows | consistently failed screening metric | only if formulation is materially changed |
| nafnet current configs | 4 ranked rows, best val R2_median 0.009407 | far below baseline | only if data interface or architecture is redesigned |
| perceiver_io current configs | 5 ranked rows, best val R2_median -0.017238 | far below baseline and unstable median | only as a redesigned explorer |
| plain AFNO current configs | 5 ranked rows, best val R2_median 0.243193 | far below baseline | only with substantial architecture or input redesign |
| UNO current configs | 5 ranked rows, best val R2_median 0.342392 | large model without val benefit | only if strong reason to revisit operator family |

## 8. Resource and implementation notes

- Known memory-heavy families: mamba2d at 423.1M params, ufno at 242.2M params, hrformer at 215.4M params, unet_afno at 188.8M params, quadmamba up to 338.4M params, residual_spectral up to 171.0M params. Current formal benchmark cap is 150M params.
- Known codegen pitfalls: Grid repair policy should stay AI-only and generated-file scoped. Do not repair shared training code from a model generation failure unless explicitly authorized.
- Required helper modules: generated standalone model wrappers under `generated_models/grid_controller_*/`. Hybrid should copy or inspect the specific wrappers for candidates it reuses.
- Models requiring special input features: `unet_sdf_7level` includes the SDF mechanism in its generated wrapper and should be reviewed before reusing the same data interface. Other rows record standard Grid gridded inputs with architecture-specific processing inside wrappers.
- Controller policy notes: smoke20 is a precheck. Benchmark200 ranked rows are the formal screening evidence. Smoke-passed candidates above 150M params are formal parameter-cap skipped, not model failures.

## 9. Files to attach or copy

- result CSV/JSON:
  - `<LOCAL_HOME_PATH>`
  - `<LOCAL_HOME_PATH>`
  - `<LOCAL_HOME_PATH>`
  - `<LOCAL_HOME_PATH>`
- model source files:
  - `<LOCAL_HOME_PATH>` through `<LOCAL_HOME_PATH>`
  - For priority reuse, inspect generated wrappers for `unet_sdf_7level`, `fourier_unet`, `unet_v3`, and `unet_v2_baseline`.
- plots: none generated in this package.
- logs:
  - `<LOCAL_HOME_PATH>` through `<LOCAL_HOME_PATH>`
- metrics folders:
  - controller final rankings under `reports/controller_grid_controller_*/final_ranking.json`

## 10. Contamination statement

- This package does not use Phase9 results, ranking, proposal rationale, reports, or hypothesis registry.
- Any overlapping architecture names with Phase9 are included only because they appear in the separate Grid controller_grid_controller_001 to controller_grid_controller_020 source session, not because of Phase9 performance.

## Appendix A. Highest ranked benchmark200 rows

| model_id | arch | params | val_R2_median | val_R2_global | val_MAE_median | round | rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grid_controller_003_05_unet_sdf_7level | unet_sdf_7level | 125,771,905 (125.8M) | 0.734949 | 0.702937 | 0.085048 | 3 | 1 |
| grid_controller_017_06_unet_sdf_7level | unet_sdf_7level | 125,771,905 (125.8M) | 0.729361 | 0.709460 | 0.086881 | 17 | 1 |
| grid_controller_018_04_unet_sdf_7level | unet_sdf_7level | 125,771,905 (125.8M) | 0.726549 | 0.710891 | 0.084685 | 18 | 1 |
| grid_controller_011_03_fourier_unet | fourier_unet | 34,201,393 (34.2M) | 0.721341 | 0.690672 | 0.083326 | 11 | 1 |
| grid_controller_019_04_unet_sdf_7level | unet_sdf_7level | 125,771,905 (125.8M) | 0.721115 | 0.700792 | 0.084222 | 19 | 1 |
| grid_controller_009_04_fourier_unet | fourier_unet | 34,201,393 (34.2M) | 0.710953 | 0.686029 | 0.087191 | 9 | 1 |
| grid_controller_019_00_unet_sdf_7level | unet_sdf_7level | 125,771,905 (125.8M) | 0.708871 | 0.692494 | 0.088401 | 19 | 2 |
| grid_controller_019_05_unet_v2_baseline | unet_v2_baseline | 34,610,193 (34.6M) | 0.708585 | 0.691738 | 0.087149 | 19 | 3 |
| grid_controller_004_05_cno_v2 | cno_v2 | 144,933,361 (144.9M) | 0.705373 | 0.684045 | 0.086512 | 4 | 1 |
| grid_controller_007_09_dilated_unet | dilated_unet | 14,780,705 (14.8M) | 0.703875 | 0.669857 | 0.088735 | 7 | 1 |
| grid_controller_012_06_multiscale_conv | multiscale_conv | 75,159,889 (75.2M) | 0.702762 | 0.687898 | 0.085647 | 12 | 1 |
| grid_controller_006_05_cno_v2 | cno_v2 | 144,933,361 (144.9M) | 0.700826 | 0.674971 | 0.090017 | 6 | 1 |



