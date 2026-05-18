#!/bin/bash
# Condor wrapper for auto_v3 train.py
# Activates conda graphwind environment, then runs train.py with config.
if [ -r /opt/crc/Modules/current/init/bash ]; then
  source /opt/crc/Modules/current/init/bash
fi
module load conda/25.9.1
source /software/c/conda/25.9.1/etc/profile.d/conda.sh
conda activate graphwind

exec python "$@"
