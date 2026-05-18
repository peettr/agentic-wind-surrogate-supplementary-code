# Baseline: reproduced Lu et al. (2025) U-Net

This folder contains the reproduced Lu-style 7-level U-Net baseline and its full reproduction pipeline.

## Main files

- `model.py`: compact baseline model entry used by the shared package.
- `config.json`: representative baseline configuration.
- `baseline_matrix_exact_summary.*`: cleaned summary metrics.
- `full_reproduction_pipeline/`: code and notes covering original data formatting, split construction, model definition, training, restoration, and evaluation.

## Full pipeline organization

1. `01_original_data_format/`: scripts for converting the original gridded wind-field/topography data into tensors.
2. `02_split_definition/`: scripts for constructing and aligning train/validation/holdout split manifests.
3. `03_model/`: Lu-style 7-level U-Net and adapter used by the shared training pipeline.
4. `04_training/`: training scripts, shared trainer, and loss definitions.
5. `05_restore/`: raw prediction restoration scripts that map model outputs back to physical/evaluation layout.
6. `06_evaluation/`: evaluation modules and baseline comparison scripts.
7. `07_run_scripts/`: representative submission/check scripts, with private paths sanitized.
8. `configs/`: representative training configs and shared config definitions.

Raw data and trained checkpoints are not included. Placeholder paths such as `<BASELINE_HPC_SOURCE_ROOT>` should be replaced by the local data location before execution.
