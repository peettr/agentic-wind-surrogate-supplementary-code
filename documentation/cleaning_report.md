# Cleaning Report

## Main cleaning and organization actions

- Removed checkpoint/model-weight files, compiled caches, logs, locks, backup files, and JSONL event streams.
- Replaced internal mode labels with public labels: Sequential, Grid, and Hybrid.
- Reorganized Grid, Sequential, and Hybrid candidate models by round, with one folder per candidate containing the model file and configuration.
- Reduced Grid to the final 200-candidate ledger: 20 rounds with 10 candidates per round.
- Removed duplicate Grid `generated_models/` source dumps; `grid_training_models/model_rounds/` is the release source of truth.
- Removed exploratory orthogonal-sweep artifacts and one unneeded Sequential round.
- Collapsed Sequential retry duplicates where a non-retry configuration exists.
- Consolidated baseline material into the common `shared_baseline/` folder used across all modes.
- Compacted prompts by keeping prompt-builder/source files, a prompt manifest, and representative full prompts only.

## Remaining manual review

Representative prompt text should still receive a final human read-through before public release.




