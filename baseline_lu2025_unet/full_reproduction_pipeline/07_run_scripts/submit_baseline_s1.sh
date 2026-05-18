#!/bin/bash
set -e
S="-o ControlPath=$HOME/.ssh/sockets/lhu1@<HPC_FILE_LOGIN>-22"
H="lhu1@<HPC_FILE_LOGIN>"
D="<BASELINE_HPC_SOURCE_ROOT>"
cd <LOCAL_WORKSPACE>/auto_v3

# Create remote dir
ssh -o BatchMode=yes -o ConnectTimeout=30 $S $H "mkdir -p $D/campaigns/baseline/runs/baseline_s1"

# Push files via stdin
cat campaigns/baseline/runs/baseline_s1/train_config.json | ssh -o BatchMode=yes -o ConnectTimeout=30 $S $H "cat > $D/campaigns/baseline/runs/baseline_s1/train_config.json" && echo config_ok
cat campaigns/baseline/runs/baseline_s1/baseline.submit | ssh -o BatchMode=yes -o ConnectTimeout=30 $S $H "cat > $D/campaigns/baseline/runs/baseline_s1/baseline.submit" && echo submit_ok

# Submit
ssh -o BatchMode=yes -o ConnectTimeout=30 $S $H "cd $D/campaigns/baseline/runs/baseline_s1 && condor_submit baseline.submit"
