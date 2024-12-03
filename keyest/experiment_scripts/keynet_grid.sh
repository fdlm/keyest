#!/usr/bin/env bash

export THEANO_FLAGS="device=${1:-cuda0}"
DRY_RUN=${2:-run}

if [ $# -gt 2 ]; then
    RUNS=${@:3}
else
    RUNS="0 1 2 3 4 5 6 7 8 9"
fi

for RUN in ${RUNS}; do
    for N_FILTERS in 8 16 24 32 40; do
        for DROPOUT in 0.0 0.1 0.2; do
            OUT=keynet_gscmbb_${N_FILTERS}_${DROPOUT}_${RUN}
            EMBEDDING_SIZE=$((${N_FILTERS} * 2))
            if ! [ -d "/home/filip/experiments/keyest/${OUT}" ]; then
                echo "========== KEYNET RUN ${RUN}   N_FILTERS ${N_FILTERS}   EMBEDDING: ${EMBEDDING_SIZE}  DROPOUT ${DROPOUT}"
                if [ "$DRY_RUN" == "run" ]; then
                    python train.py \
                        --model KeyNet --data giantsteps,billboard,cmdb \
                        --model_params "{l2: 0.0, n_filters: ${N_FILTERS}, embedding_size: ${EMBEDDING_SIZE}, dropout: ${DROPOUT}}" \
                        --out ${OUT}
                fi
            fi
        done
    done
done
