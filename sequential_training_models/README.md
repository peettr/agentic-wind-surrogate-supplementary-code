# Sequential Training and Models

This folder contains Sequential training/evaluation code and round-organized models.

The release source of truth for candidate files is `model_rounds/`. Each round contains one subfolder per candidate, and each candidate folder contains `model.py` and `config.json`. Retry duplicates were collapsed in favor of the non-retry configuration when both existed, and the unneeded removed_round round was removed.






## Release organization note

The canonical model artifacts for Sequential are in model_rounds/, with each candidate stored as an independent folder containing model.py and config.json. Internal controller/orchestration scripts and duplicate shared model registries are intentionally excluded from the public package; the remaining shared/ files provide common training, evaluation, loss, and configuration support.

