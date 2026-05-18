#!/usr/bin/env bash
set -euo pipefail
if [ -r /opt/crc/Modules/current/init/bash ]; then
  source /opt/crc/Modules/current/init/bash
fi
module load conda/25.9.1
source /software/c/conda/25.9.1/etc/profile.d/conda.sh
conda activate graphwind
cd <BASELINE_HPC_SOURCE_ROOT>
python <GRID_HPC_SOURCE_ROOT>/reports/holdout_eval_auto11_top/eval_grid18_baseline_holdout.py
