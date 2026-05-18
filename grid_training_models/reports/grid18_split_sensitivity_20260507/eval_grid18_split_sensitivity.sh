#!/bin/bash
set -euo pipefail
if [ -r /opt/crc/Modules/current/init/bash ]; then
  source /opt/crc/Modules/current/init/bash
fi
module load conda/25.9.1 || true
source /software/c/conda/25.9.1/etc/profile.d/conda.sh
conda activate graphwind
cd <PROJECT_HPC_ROOT>
python3 <GRID_HPC_SOURCE_ROOT>/reports/grid18_split_sensitivity_20260507/eval_grid18_split_sensitivity.py
