# Grid Remote Code Pull Addendum

Pulled Grid training/model code from CRC source root:

`<GRID_HPC_SOURCE_ROOT>`

Copied into:

- `grid_training_models/shared/`
- `grid_training_models/scripts/`
- `grid_training_models/templates/`
- `grid_training_models/generated_models/`

Excluded cache, checkpoint, and log files. Campaign run directories were not copied because the remote `campaigns/` tree is approximately 212 GB and contains large run artifacts. Cleaned summaries and report artifacts are included separately from the local report package.
