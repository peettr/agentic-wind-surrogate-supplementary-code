# Prompts and Prompt Generators

This directory provides a compact, release-oriented prompt record.

## Contents

- `prompt_templates_and_builders/`: source files from the Hybrid workflow that construct or contain prompts.
- `representative_full_prompts/`: selected full prompts from representative Hybrid rounds.
- `prompt_manifest.csv`: inventory of prompt artifacts from the original workflow archive, including size and whether the full prompt is retained.

## Rationale

The original workflow stored every full prompt for every round. Late-round planner prompts contained accumulated context and could exceed 10 MB each, with substantial repetition across rounds. For release, the package keeps representative full prompts and the prompt-building code/templates needed to understand and reconstruct the prompting process. Raw JSON/JSONL provider streams and all non-representative full prompt copies are excluded.
