# Auto_v5 Protocol

## Purpose

This document defines the main Auto_v5 benchmark protocol derived from Auto_v3 findings. It separates hard rules from soft rules and is written as an operational instruction set for agents and humans.

The goal is to keep Auto_v5 comparable, auditable, and compute-efficient by fixing the task and evaluation contracts while focusing search on the variables that mattered in Auto_v3.

---

## Hard Rules, mandatory

These rules are mandatory for any run that will be included in the main Auto_v5 benchmark. If a run violates any item below, it must be labeled exploratory and excluded from the main leaderboard.

### Task and data contract

1. All main benchmark runs must use the fixed 640×640 task contract.
2. All main benchmark runs must use the same locked split manifest.
3. Validation and holdout definitions must remain fixed across all compared runs.
4. Within a given benchmark round, the input representation must be identical for all compared models.
5. Different input representations must not be mixed in the same main leaderboard.

### Evaluation contract

6. All official results must be produced with the official evaluation module.
7. Simplified tensor-level evaluation must not replace official evaluation.
8. The primary ranking metric is per-case median R².
9. Each official result must also report MAE, global R², wall time, and peak VRAM.
10. All main benchmark runs must set `compute_r2=true`.

### Training contract

11. All main benchmark runs must use `epochs=200`.
12. All main benchmark runs must satisfy `batch_size<=16`.
13. The default main benchmark batch size is `16`.
14. All main benchmark runs must set `data_augment=false`.
15. All main benchmark runs must follow the same logging, checkpoint, and result export conventions.

### Resource contract

16. All main benchmark candidates must satisfy `params<=150M`.
17. All main benchmark candidates must be runnable within `<=40GB VRAM`.
18. Models that exceed either limit must be labeled exploratory and excluded from the main benchmark leaderboard.

### Auditability and experiment organization

19. Every run must save `train_config.json`, `metrics.json`, `train.log`, hardware information, and an execution record.
20. Missing artifacts disqualify a run from serving as formal benchmark evidence.
21. Training recipe search and architecture benchmark must be separated into distinct phases.
22. Do not run a full hyperparameter sweep separately for every architecture.
23. Exploratory runs must never be merged into the main leaderboard.

---

## Soft Rules, default policy

These are default policies for the main Auto_v5 workflow. They may be changed only with explicit justification recorded in the experiment note or report.

### Recommended search variables for Phase 1

1. Restrict the first-stage loss search to:
   - `masked_l1`
   - `masked_l1_gradient`
2. Restrict the first-stage learning rate search to:
   - `1e-3`
   - `5e-4`
3. If EMA is confirmed to use the bug-fixed implementation, restrict EMA search to:
   - `None`
   - `0.999`
4. If scheduler is included, restrict it to:
   - `None`
   - `cosine`

### Variables excluded from first-stage main search by default

Do not include the following in the first-stage main search unless a separate justification is provided:

- `dropout`
- `activation`
- `norm_type`
- generic `data_augment`
- broad weight decay sweeps
- broad grad clip sweeps

### Recommended three-phase structure

#### Phase 1, recipe search

Use one representative architecture and a fixed input representation.

Recommended search variables:
- `loss ∈ {masked_l1, masked_l1_gradient}`
- `lr ∈ {1e-3, 5e-4}`
- `ema ∈ {None, 0.999}`
- optional `scheduler ∈ {None, cosine}`

Fixed variables:
- `epochs=200`
- `batch_size=16`
- `compute_r2=true`
- `data_augment=false`
- locked split
- fixed input representation

#### Phase 2, architecture benchmark

Select the top 1 to 2 recipes from Phase 1 and apply them uniformly to all candidate architectures.

Default policy:
- each architecture should receive at most 1 to 2 main benchmark recipes
- do not customize a large hyperparameter sweep per architecture

#### Phase 3, finalist refinement

Only top-performing models may enter finalist refinement.

Optional variables that may be opened here:
- `scheduler`
- `weight_decay`
- `grad_clip`
- longer training
- multi-seed verification

### Recommended default recipes

#### Recipe A, robust default

- `loss = masked_l1_gradient`
- `lr = 1e-3`
- `ema = None`
- `scheduler = cosine` or `None`, choose one before launch
- `batch_size = 16`
- `epochs = 200`
- `data_augment = false`

#### Recipe B, performance-oriented

- `loss = masked_l1`
- `lr = 5e-4`
- `ema = 0.999`
- `scheduler = None`
- `batch_size = 16`
- `epochs = 200`
- `data_augment = false`

### Final validation recommendation

After the main benchmark, only the top 1 to 2 models should enter final validation.

Recommended final validation policy:
- `epochs=1000`
- multi-seed evaluation
- official evaluation only
- reported separately from the 200-epoch benchmark table

---

## Operational Instruction Set for Agents

Use the following instructions when executing Auto_v5 experiments.

### Mandatory execution rules

1. Use the fixed 640×640 task contract.
2. Use the locked split manifest shared by all compared runs.
3. Use the official evaluation module for every official metric report.
4. Set `compute_r2=true` for all main benchmark runs.
5. Set `epochs=200` for all main benchmark runs.
6. Enforce `batch_size<=16`, default `batch_size=16`.
7. Set `data_augment=false`.
8. Include only candidates with `params<=150M` in the main benchmark.
9. Include only candidates runnable within `<=40GB VRAM` in the main benchmark.
10. Save full run artifacts for every run.
11. Separate recipe search from architecture benchmark.
12. Never mix exploratory runs into the main leaderboard.

### Default execution rules

1. In Phase 1, search only `loss`, `lr`, `ema`, and optional `scheduler`.
2. In Phase 1, prioritize `masked_l1`, `masked_l1_gradient`, `1e-3`, and `5e-4`.
3. Exclude `dropout`, `activation`, `norm_type`, and generic augmentation from first-stage search.
4. In Phase 2, apply the top 1 to 2 fixed recipes uniformly to all architectures.
5. In Phase 3, refine only the top models.

### Required reporting format

Every status update and final report must state:

- current phase, Phase 1, Phase 2, or Phase 3
- whether the run fully complies with hard rules
- any deviation from soft rules and why it was made
- whether a run is benchmark or exploratory
- the exact recipe used
- the exact resource class used

If any hard rule is violated, the run must be labeled exploratory immediately.

---

## Short Form

### Hard rules

- fixed 640×640 task contract
- fixed split manifest
- official evaluation only
- `compute_r2=true`
- `epochs=200`
- `batch_size<=16`
- `data_augment=false`
- `params<=150M`
- `VRAM<=40GB`
- full artifact retention
- recipe search separated from architecture benchmark
- no exploratory runs in the main leaderboard

### Soft rules

- first-stage search on `loss`, `lr`, `ema`, optional `scheduler`
- prioritize `masked_l1`, `masked_l1_gradient`
- prioritize `1e-3`, `5e-4`
- default EMA choices are `None` and `0.999`
- exclude `dropout`, `activation`, `norm_type`, and generic augmentation from first-stage search
- benchmark each architecture with at most 1 to 2 fixed recipes
- reserve long training and multi-seed validation for finalists only
