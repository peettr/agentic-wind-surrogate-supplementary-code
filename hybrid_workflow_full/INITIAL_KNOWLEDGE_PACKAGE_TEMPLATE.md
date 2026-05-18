# Hybrid Initial Knowledge Package Template

> Purpose: provide Hybrid with experimental evidence from a separate session. Do not include any Phase9 result, ranking, report, hypothesis, or proposal rationale.

## 0. Provenance

- Source session / project:
- Authoring agent:
- Date:
- Data source:
- Split used:
- Training/evaluation protocol:
- Any known deviations from Hybrid contract:

## 1. Executive summary

Briefly state what was learned from the separate experiment session.

Recommended length: 5-10 bullets.

## 2. Baseline and metric definitions

- Search metric:
- Validation split size:
- Holdout/test split size, if any:
- Baseline model:
- Baseline metric values:
- Whether results are directly comparable to Hybrid:

## 3. Full model result table

Use one row per trained/evaluated model. Include failed runs if they inform feasibility.

| model_id | family | arch_name | params | trainable_params | input_features | n_c/depth/width | lr | loss | epochs | seed | val_R2_median | val_R2_mean | val_R2_global | val_MAE_median | holdout_R2_median | status | notes |
|---|---|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|

## 4. Per-family synthesis

For each architecture family or mechanism:

### Family / mechanism name

- Best model_id:
- Main result:
- Parameter efficiency:
- Failure modes:
- What seems to work:
- What seems not to work:
- Confidence level: high / medium / low
- Suggested Hybrid action: exploit / ablate / control / avoid / explore further

## 5. Hypotheses for Hybrid

Each hypothesis should be actionable and testable.

### H-001: Short title

- Claim:
- Evidence from source session:
- Proposed Hybrid test:
- Paired comparison / control:
- Decision rule:
- Expected success:
- Expected failure interpretation:
- Risk / resource expectation:

## 6. Candidate seeds for Hybrid planner

These are not mandatory configs, but planner guidance.

| priority | role | mechanism | suggested config sketch | paired control | rationale | avoid conditions |
|---:|---|---|---|---|---|---|

Roles: exploit, ablation, control, explorer.

## 7. Negative knowledge / avoid list

List configs or mechanisms that should not be repeated unless there is a specific reason.

| item | evidence | reason to avoid | exception condition |
|---|---|---|---|

## 8. Resource and implementation notes

- Known memory-heavy families:
- Known codegen pitfalls:
- Required helper modules:
- Models requiring special input features:
- Models requiring special loss/eval handling:

## 9. Files to attach or copy

List files that Hybrid should read or copy.

- result CSV/JSON:
- model source files:
- plots:
- logs:
- metrics folders:

## 10. Contamination statement

Confirm explicitly:

- This package does not use Phase9 results, ranking, proposal rationale, reports, or hypothesis registry.
- Any overlapping architecture names with Phase9 are included only because they appear in the separate source session, not because of Phase9 performance.



