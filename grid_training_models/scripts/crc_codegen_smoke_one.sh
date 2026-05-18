#!/usr/bin/env bash
set -euo pipefail

CTL="${CRC_CONTROL_PATH:-<SSH_CONTROL_PATH>"
HOST="${CRC_HOST:-<HPC_USER>@<HPC_LOGIN>}"
LOCAL="<LOCAL_HOME_PATH>"
REMOTE="${REMOTE:-<BASELINE_HPC_SOURCE_ROOT>}"
CAMPAIGN="${CAMPAIGN:-v5_codegen_smoke_test_001}"
RUN_ID="${RUN_ID:-r_codegen_unet_v2_baseline_smoke20}"
MODEL_FILE="${MODEL_FILE:-generated_models/v5_codegen_test_001/unet_v2_baseline.py}"
MODULE_NAME="${MODULE_NAME:-codegen_unet_v2_baseline}"
STAGE="${STAGE:-smoke20}"
USER_BUNDLE="/users/lhu1/${CAMPAIGN}"
LOG_ROOT="/users/lhu1/condor_v5_logs"
SUBMIT_TIER="${SUBMIT_TIER:-a40_rtx6k_16gb}"
TAR="$LOCAL/tmp_${CAMPAIGN}.tar"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=120 -o ServerAliveInterval=10 -o ControlPath=$CTL $HOST"
SCP="scp -o BatchMode=yes -o ConnectTimeout=120 -o ControlPath=$CTL"

case "$SUBMIT_TIER" in
  h100_only_12gb|h100_a100_l40s_12gb|a40_rtx6k_16gb|a10_16gb) ;;
  *) echo "Unknown SUBMIT_TIER=$SUBMIT_TIER" >&2; exit 2 ;;
esac

cd "$LOCAL"

echo "=== Generate tiered Condor submit files ==="
python scripts/validate_train_configs.py \
  --campaign "campaigns/$CAMPAIGN" \
  --stage "$STAGE" \
  --run-id "$RUN_ID"
python scripts/generate_tiered_condor_submits.py \
  --campaign-name "$CAMPAIGN" \
  --run-id "$RUN_ID" \
  --remote-root "$REMOTE" \
  --log-root "$LOG_ROOT" \
  --wrapper-path "$USER_BUNDLE/condor_wrapper.sh" \
  --output-dir "campaigns/$CAMPAIGN"

# Keep the legacy submit filename as the selected tier alias for manual use.
cp "campaigns/$CAMPAIGN/codegen_smoke_${SUBMIT_TIER}.submit" "campaigns/$CAMPAIGN/codegen_smoke.submit"

echo "=== Pack Auto V5 smoke test code ==="
tar --exclude='__pycache__' --exclude='.pytest_cache' --exclude='shared/data' -cf "$TAR" \
  shared scripts templates generated_models \
  campaigns/$CAMPAIGN AUTO_V5_PROTOCOL.md

echo "=== Create remote project root, user wrapper, and log dir ==="
$SSH "mkdir -p '$REMOTE' '$USER_BUNDLE' '$LOG_ROOT/$CAMPAIGN/$RUN_ID'"

echo "=== Upload tar and wrapper ==="
$SCP "$TAR" "$HOST:$REMOTE/tmp_${CAMPAIGN}.tar"
$SCP "$LOCAL/templates/condor_wrapper.sh" "$HOST:$USER_BUNDLE/condor_wrapper.sh"

$SSH bash -s <<EOF
set -euo pipefail
REMOTE="$REMOTE"
CAMPAIGN="$CAMPAIGN"
RUN_ID="$RUN_ID"
USER_BUNDLE="$USER_BUNDLE"
LOG_ROOT="$LOG_ROOT"
cd "\$REMOTE"
tar xf "tmp_\${CAMPAIGN}.tar"
rm -f "tmp_\${CAMPAIGN}.tar"
mkdir -p shared "\$LOG_ROOT/\$CAMPAIGN/\$RUN_ID"
if [ "\$REMOTE" = "<BASELINE_HPC_SOURCE_ROOT>" ]; then
  if [ -L shared/data ] && [ "\$(readlink shared/data)" = "\$REMOTE/shared/data" ]; then
    rm -f shared/data
    mkdir -p shared/data
  else
    mkdir -p shared/data
  fi
else
  rm -rf shared/data
  ln -s <BASELINE_HPC_SOURCE_ROOT>/shared/data shared/data
fi
chmod 755 "\$USER_BUNDLE/condor_wrapper.sh"
source /opt/crc/Modules/current/init/bash 2>/dev/null || true
module load conda/25.9.1 2>/dev/null || true
source /software/c/conda/25.9.1/etc/profile.d/conda.sh
conda activate graphwind
python - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, '$REMOTE')
from shared.configs.schema import TrainConfig
cfg_path = Path('$REMOTE') / 'campaigns' / '$CAMPAIGN' / 'runs' / '$RUN_ID' / 'train_config.json'
TrainConfig.model_validate(json.loads(cfg_path.read_text()))
print('REMOTE_TRAIN_CONFIG_SCHEMA_OK', cfg_path)
PY
rm -f "\$LOG_ROOT/\$CAMPAIGN/\$RUN_ID"/condor.out "\$LOG_ROOT/\$CAMPAIGN/\$RUN_ID"/condor.err "\$LOG_ROOT/\$CAMPAIGN/\$RUN_ID"/condor.log
RUN_DIR="\$REMOTE/campaigns/\$CAMPAIGN/runs/\$RUN_ID"
rm -f "\$RUN_DIR"/train.log "\$RUN_DIR"/FAILED "\$RUN_DIR"/FINISHED "\$RUN_DIR"/HEARTBEAT.json "\$RUN_DIR"/metrics.json
EOF

echo "=== Frontend graphwind dynamic import/forward check ==="
$SSH bash -s <<EOF
set -euo pipefail
source /opt/crc/Modules/current/init/bash 2>/dev/null || true
module load conda/25.9.1 2>/dev/null || true
source /software/c/conda/25.9.1/etc/profile.d/conda.sh
conda activate graphwind
python - <<PY
import sys, importlib.util, torch, os
from pathlib import Path
base=Path('$REMOTE')
sys.path.insert(0, str(base))
p=base/'$MODEL_FILE'
spec=importlib.util.spec_from_file_location('$MODULE_NAME', p)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
m=mod.Model(in_channels=1, out_channels=1)
m.eval()
x=torch.randn(1,1,128,128)
with torch.no_grad():
    y=m(x)
params=sum(p.numel() for p in m.parameters())
limit=int(os.environ.get('AUTO_V5_PARAM_LIMIT', '150000000'))
print('FRONTEND_DYNAMIC_OK', tuple(y.shape), 'params', params, 'limit', limit)
if tuple(y.shape) != (1, 1, 128, 128):
    raise SystemExit(f'shape contract failed: {tuple(y.shape)}')
if params > limit:
    raise SystemExit(f'parameter limit exceeded: {params} > {limit}')
PY
EOF

echo "=== Submit Condor smoke test tier: $SUBMIT_TIER ==="
$SSH "cd '$REMOTE' && condor_submit 'campaigns/$CAMPAIGN/codegen_smoke_${SUBMIT_TIER}.submit'"

echo "=== Query jobs ==="
$SSH "condor_q lhu1 -name <HPC_FILE_LOGIN> -nobatch | tail -20"
