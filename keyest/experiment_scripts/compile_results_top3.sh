#!/usr/bin/env bash

EXPS="allconv_gscmbb_20_0.1_1 allconv_gscmbb_24_0.0_2 allconv_gscmbb_24_0.1_2 allconv_gscmbb_20_0.1_9 allconv_gscmbb_20_0.0_6"

for EXP in ${EXPS}; do
    echo "=================== ${EXP}"
    python test.py --exp_dir /home/filip/experiments/keyest/${EXP} --data giantsteps,billboard,cmdb --save_pred --proc_aug
done
