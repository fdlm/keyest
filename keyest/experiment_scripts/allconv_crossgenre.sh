#!/usr/bin/env bash

export THEANO_FLAGS="device=${1:-cuda0}"
DRY_RUN=${2:-run}

if [ $# -gt 2 ]; then
    RUNS=${@:3}
else
    RUNS="0 1 2 3 4 5 6 7 8 9"
fi

declare -A DATASETS
DATASETS[gs]=giantsteps
DATASETS[bb]=billboard
DATASETS[cm]=cmdb
DATASETS[gsbb]=${DATASETS[gs]},${DATASETS[bb]}
DATASETS[gscm]=${DATASETS[gs]},${DATASETS[cm]}
DATASETS[cmbb]=${DATASETS[cm]},${DATASETS[bb]}

for RUN in ${RUNS}; do
    for N_FILTERS in 8 20; do
        for DATACFG in gs bb cm gsbb gscm cmbb; do
            if [ ${N_FILTERS} -gt 8 ]; then
                DROPOUT=0.1
            else
                DROPOUT=0.0
            fi
            OUT=allconv_${DATACFG}_${N_FILTERS}_${DROPOUT}_${RUN}
            if ! [ -d "/home/filip/experiments/keyest/${OUT}" ]; then
                echo "========== ALLCONV RUN ${RUN}   DATA ${DATACFG}   N_FILTERS ${N_FILTERS}   DROPOUT ${DROPOUT}"
                if [ "$DRY_RUN" == "run" ]; then
                    python train.py \
                        --model AllConv --data ${DATASETS[${DATACFG}]} \
                        --model_params "{l2: 0.0, n_filters: ${N_FILTERS}, dropout: ${DROPOUT}}" \
                        --out ${OUT}
                fi
            fi
        done
    done
done
