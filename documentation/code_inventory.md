# Code Inventory

## Shared baseline

- `shared_baseline/model.py`: reproduced Lu et al. (2025) U-Net baseline implementation.
- `shared_baseline/config.json`: representative baseline training configuration.
- `shared_baseline/full_reproduction_pipeline/`: full baseline reproduction workflow, including original data formatting, split construction, training, restoration, and evaluation code.

## Hybrid

- `hybrid_workflow_full/`: complete workflow implementation for the primary Hybrid mode.
- `hybrid_training_models/model_rounds/`: Hybrid full-run candidates organized by round, with one subfolder per candidate containing `model.py`, `config.json`, and available attempt metadata.
- `hybrid_training_models/shared/`: shared training, evaluation, losses, configs, and utility code.

## Sequential

- `sequential_training_models/model_rounds/`: Sequential candidates organized by round, with one subfolder per candidate containing `model.py` and `config.json`.
- `sequential_training_models/shared/`: Sequential shared training and evaluation code.

## Grid

- `grid_training_models/model_rounds/`: final 200 Grid candidates, organized as 20 rounds with 10 candidate subfolders per round. Each candidate contains `model.py` and `config.json`.
- `grid_training_models/model_rounds_manifest.csv`: round-level model inventory.
- `grid_training_models/shared/`: Grid shared training/evaluation/config code.
- `grid_training_models/scripts/`: Grid run/evaluation scripts retained for reproducibility.

## Prompts and results

- `prompts_and_prompt_generators/`: prompt templates/builders, representative prompts, and manifest.
- `results_summary/`: cleaned summary result files.

## Excluded

- Checkpoints and model weights.
- Raw large campaign run directories.
- Duplicate generated-model source dumps outside `model_rounds/`.
- Exploratory orthogonal-sweep artifacts.
- Cache files, compiled files, logs, locks, backup files, and JSONL event streams.
- Private channel notification scripts and provider streams.




