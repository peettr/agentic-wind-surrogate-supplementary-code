# Code Inventory Addendum

Additional deployment step after initial draft assembly:

- Copied Hybrid artifact prompt/review/scout/codegen/planner/quality-gate files from `<HYBRID_SOURCE_ROOT>\campaigns\auto_v6\artifacts` into `prompts_and_prompt_generators/hybrid_artifact_prompts_raw/`.
- These prompt artifacts are raw and must be sanitized before public release.
- Grid full source is currently represented by the local `auto_v5_report_package`; if the original remote Grid source is required for exact training/model code beyond the package, pull it from the manifest source root before final release.
