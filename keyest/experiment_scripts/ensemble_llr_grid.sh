#!/usr/bin/env bash

MODELS=(allconv_gscmbb_24_0.1_2 allconv_gscmbb_20_0.1_9 allconv_gscmbb_24_0.0_2 \
        allconv_gscmbb_20_0.1_0 allconv_gscmbb_20_0.1_1 allconv_gscmbb_12_0.0_3 \
        allconv_gscmbb_12_0.1_7 allconv_gscmbb_20_0.2_4 allconv_gscmbb_24_0.1_1 \
        allconv_gscmbb_24_0.1_7)

for I in ${!MODELS[@]}; do
    N_MODELS=$((I + 1))
    ENSEMBLE_MODELS=${MODELS[*]::${N_MODELS}}
    python ensemble/tag_informed_llr.py --simple --data giantsteps,billboard,cmdb \
           --train_on_val --out ensemble_llr_${N_MODELS} \
           ${ENSEMBLE_MODELS}
done
