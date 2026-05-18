# Reproduced Lu Baseline Pipeline Overview

The baseline pipeline follows this sequence.

1. Convert original wind-field and topography files into tensor datasets using the formatter scripts in `01_original_data_format/`.
2. Construct the split manifest using `02_split_definition/`, preserving the aligned validation/holdout definitions used in the reported experiments.
3. Train the Lu-style 7-level U-Net using `04_training/` and the representative configs in `configs/`.
4. Restore raw model predictions to the evaluation layout with `05_restore/step2_raw_restore.py`.
5. Evaluate restored predictions with the scripts in `06_evaluation/`, using the metric definitions shared with the Sequential, Grid, and Hybrid experiments.

The code is provided for reproducibility and auditability. Raw data and checkpoint weights are intentionally excluded from the code package.
