# Cleaning Report

Clean package generated from `supplementary_code_draft` and then reorganized for release.

## Main cleaning and organization actions

- Removed checkpoint/model-weight files, compiled caches, logs, locks, backup files, and JSONL event streams.
- Removed raw JSON/JSONL prompt streams.
- Compacted prompts by keeping prompt-builder/source files, a prompt manifest, and representative full prompts only.
- Removed private channel notification scripts.
- Replaced local user paths, raw HPC project paths, SSH socket paths, and known private literals with placeholders.
- Limited Grid to 20 complete 10-candidate rounds, 200 models total.
- Added round-organized `model_rounds/` directories for Grid, Sequential, and Hybrid.
- Expanded the Lu et al. (2025) baseline folder with a full reproduction pipeline covering data formatting, split construction, training, restoration, and evaluation.

## Remaining manual review

Representative prompt text should still receive a final human read-through before public release. The current package favors a compact, readable prompt record rather than a full duplicated prompt archive.

