#!/usr/bin/env bash

MODELS=(allconv_gscmbb_24_0.1_2 allconv_gscmbb_20_0.1_9 allconv_gscmbb_24_0.0_2 \
        allconv_gscmbb_20_0.1_0 allconv_gscmbb_20_0.1_1 allconv_gscmbb_12_0.0_3 \
        allconv_gscmbb_12_0.1_7 allconv_gscmbb_20_0.2_4 allconv_gscmbb_24_0.1_1 \
        allconv_gscmbb_24_0.1_7)

for I in ${!MODELS[@]}; do
    N_MODELS=$((I + 1))
    ENSEMBLE_MODELS=${MODELS[*]::${N_MODELS}}
    for TEMP in 0.1 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0; do
        python ensemble/avg_ensemble.py --temperature ${TEMP} \
               ensemble_avg_${N_MODELS}_T=${TEMP} ${ENSEMBLE_MODELS}
    done
done
