#!/usr/bin/env bash

EXPS='allconv_gscmbb_20_0.0_6 allconv_cm_20_0.1_3 allconv_cm_20_0.1_1 allconv_gscmbb_16_0.1_8 allconv_gscm_20_0.1_6 allconv_gscmbb_20_0.2_2 allconv_cm_8_0.0_5 allconv_gscmbb_20_0.1_9 allconv_gscmbb_20_0.2_4 allconv_gscmbb_24_0.0_2 allconv_gscmbb_24_0.1_2 allconv_gscmbb_20_0.1_1 allconv_gscmbb_24_0.2_3 allconv_gscmbb_24_0.2_0'

for EXP in ${EXPS}; do
    echo "=================== ${EXP}"
    python test.py --exp_dir /home/filip/experiments/keyest/${EXP} --data giantsteps,billboard,cmdb --save_pred
done
