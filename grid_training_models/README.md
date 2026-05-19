# Grid Training and Models

This folder contains the final 200 Grid candidates, organized as 20 rounds with 10 model folders per round.

`generated_models/` was removed to avoid duplicating the same files. The release source of truth is `model_rounds/`; each candidate folder contains `model.py` and `config.json`.

orthogonal exploratory sweep exploratory artifacts and earlier controller/source-dump directories are not included.






## Release organization note

The canonical model artifacts for Grid are in model_rounds/, with each candidate stored as an independent folder containing model.py and config.json. Internal controller/orchestration scripts and duplicate shared model registries are intentionally excluded from the public package; the remaining shared/ files provide common training, evaluation, loss, and configuration support.

