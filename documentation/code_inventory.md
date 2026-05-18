# Code Inventory

## Baseline

- `baseline_lu2025_unet/model.py`: reproduced Lu et al. (2025) U-Net baseline implementation.
- `baseline_lu2025_unet/config.json`: representative baseline training configuration.
- `baseline_lu2025_unet/baseline_matrix_exact_summary.*`: cleaned summary metrics.
- `baseline_lu2025_unet/full_reproduction_pipeline/`: full baseline reproduction workflow, including original data formatting, split construction, training, restoration, and evaluation code.

## Hybrid

- `hybrid_workflow_full/`: complete workflow implementation for the primary Hybrid mode.
- `hybrid_training_models/models/`: Hybrid model implementations.
- `hybrid_training_models/shared/`: shared training, evaluation, losses, configs, and utility code.
- `hybrid_training_models/model_rounds/`: Hybrid full-run candidates organized by round, with each experiment carrying its own `model.py`, `config.json`, and available attempt metadata.
- `hybrid_training_models/model_rounds_manifest.csv`: round-level model inventory.

## Sequential

- `sequential_training_models/models/`: Sequential model implementations.
- `sequential_training_models/shared/`: Sequential shared training and evaluation code.
- `sequential_training_models/selected_round_configs/`: full training configurations by round.
- `sequential_training_models/model_rounds/`: Sequential candidates organized by round, with each experiment carrying its own `model.py` and `config.json`.
- `sequential_training_models/model_rounds_manifest.csv`: round-level model inventory.

## Grid

- `grid_training_models/generated_models/`: final Grid generated model directories only, 20 complete 10-candidate rounds, `v5_controller_auto10_006` through `v5_controller_auto10_008` and `v5_controller_auto11_001` through `v5_controller_auto11_017`.
- `grid_training_models/model_rounds/`: the same final Grid candidates reorganized as 20 rounds, 10 candidate models per round.
- `grid_training_models/model_rounds_manifest.csv`: round-level model inventory.
- `grid_training_models/shared/`: Grid shared training/evaluation/config code.
- `grid_training_models/scripts/`: Grid run/evaluation scripts.
- `grid_training_models/metadata/`: Grid definitions and candidate metadata.

## Prompts

- `prompts_and_prompt_generators/prompt_templates_and_builders/`: source files from the Hybrid workflow that construct or contain prompts.
- `prompts_and_prompt_generators/representative_full_prompts/`: representative full prompts from selected Hybrid rounds.
- `prompts_and_prompt_generators/prompt_manifest.csv`: inventory of original prompt artifacts and retained representative files.

## Results

- `results_summary/hybrid_corrected_val55/`: Hybrid corrected validation selection summaries.
- `results_summary/sequential_summary/`: Sequential reporting summaries.
- `results_summary/grid_summary/`: Grid reporting summaries.

## Excluded

- Checkpoints and model weights.
- Raw large campaign run directories.
- Cache files, compiled files, logs, locks, backup files, and JSONL event streams.
- Private channel notification scripts.
- Raw model provider streams and transient JSON/JSONL prompt outputs.
- Non-representative duplicated full prompt artifacts.
- Earlier Grid exploratory/codegen/test rounds outside the final 20 Grid rounds.

