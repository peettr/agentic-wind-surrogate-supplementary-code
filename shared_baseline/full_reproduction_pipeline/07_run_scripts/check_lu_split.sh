#!/bin/bash
S="-o ControlPath=$HOME/.ssh/sockets/lhu1@<HPC_FILE_LOGIN>-22"
H="lhu1@<HPC_FILE_LOGIN>"
D="<PROJECT_HPC_ROOT>"
ssh $S $H "ls $D/auto_v2/full_dataset/scripts/all_cases_20exp/ | head -30 && echo '---' && head -3 $D/auto_v2/full_dataset/scripts/all_cases_20exp/metrics_in_training_set_seed1.csv && echo '---' && wc -l $D/auto_v2/full_dataset/scripts/all_cases_20exp/metrics_in_training_set_seed1.csv $D/auto_v2/full_dataset/scripts/all_cases_20exp/metrics_in_test_set_seed1.csv"



