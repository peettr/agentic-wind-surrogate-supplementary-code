# Supplementary Code Package

This package contains source code and supporting artifacts for the baseline model and the three search modes reported in the study: Sequential, Grid, and Hybrid.

## Contents

- `baseline_lu2025_unet/`: reproduced Lu et al. (2025) U-Net baseline.
- `hybrid_workflow_full/`: complete Hybrid workflow implementation.
- `hybrid_training_models/`: Hybrid training, evaluation, loss, configuration, and model source code.
- `sequential_training_models/`: Sequential training, evaluation, configuration, and model source code.
- `grid_training_models/`: Grid training, evaluation, generated model, configuration, and reporting source code.
- `prompts_and_prompt_generators/`: prompt templates/builders, representative prompts, and prompt manifest.
- `results_summary/`: cleaned summary result files.
- `documentation/`: code inventory, cleaning report, and release checklist.

Large checkpoints, raw run directories, caches, logs, socket paths, and private notification scripts are excluded.
