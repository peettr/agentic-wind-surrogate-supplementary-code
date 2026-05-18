# Supplementary Code Package

This package contains source code and supporting artifacts for the shared baseline and the three search modes reported in the study: Sequential, Grid, and Hybrid.

## Contents

- `shared_baseline/`: common reproduced Lu et al. (2025) U-Net baseline, including the full reproduction pipeline from original data formatting and split definition through training, restoration, and evaluation.
- `hybrid_workflow_full/`: complete Hybrid workflow implementation.
- `hybrid_training_models/`: Hybrid training/evaluation code plus models organized by round in `model_rounds/`.
- `sequential_training_models/`: Sequential training/evaluation code plus models organized by round in `model_rounds/`.
- `grid_training_models/`: final 200 Grid candidates, organized as 20 rounds with 10 model folders per round.
- `prompts_and_prompt_generators/`: prompt templates/builders, representative prompts, and prompt manifest.
- `results_summary/`: cleaned summary result files.
- `documentation/`: code inventory, cleaning report, and release checklist.

Large checkpoints, raw run directories, duplicate source dumps, exploratory orthogonal exploratory sweep artifacts, caches, logs, socket paths, and private notification scripts are excluded.



