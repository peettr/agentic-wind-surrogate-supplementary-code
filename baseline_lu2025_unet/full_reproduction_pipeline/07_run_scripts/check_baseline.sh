#!/bin/bash
S="-o ControlPath=$HOME/.ssh/sockets/lhu1@<HPC_FILE_LOGIN>-22"
H="lhu1@<HPC_FILE_LOGIN>"
D="<BASELINE_HPC_SOURCE_ROOT>/campaigns/baseline/runs/baseline_s1"
ssh -o BatchMode=yes -o ConnectTimeout=30 $S $H "condor_q 17759 2>&1 | head -6; echo '==='; tail -10 $D/condor.out 2>/dev/null; echo '===err==='; tail -5 $D/condor.err 2>/dev/null"
