#!/usr/bin/env bash

export THEANO_FLAGS="device=${1:-cuda0}"
DRY_RUN=${2:-run}

for RUN in {0..9}; do
    for N_FILTERS in 2 4 8 12 16 20; do
        for DROPOUT in 0.0 0.1 0.2; do
            OUT=allconv_gscmbb_${N_FILTERS}_${DROPOUT}_${RUN}
            if ! [ -d "/home/filip/experiments/keyest/${OUT}" ]; then
                echo "========== ALLCONV RUN ${RUN}   N_FILTERS ${N_FILTERS}   DROPOUT ${DROPOUT}"
                if [ "$DRY_RUN" == "run" ]; then
                    python train.py \
                        --model AllConv --data giantsteps,billboard,cmdb \
                        --model_params "{l2: 0.0, n_filters: ${N_FILTERS}, dropout: ${DROPOUT}}" \
                        --out ${OUT}
                fi
            fi
        done
    done
done
