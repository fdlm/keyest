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
DATASETS[gscmbb]=${DATASETS[gs]},${DATASETS[cm]},${DATASETS[bb]}

for RUN in ${RUNS}; do
    N_FILTERS=20
    DATACFG=gscmbb
    DROPOUT=0.1
    OUT=allconv_${DATACFG}_${N_FILTERS}_${DROPOUT}_tagweighted_${RUN}
    if ! [ -d "/home/filip/experiments/keyest/${OUT}" ]; then
        echo "========== ALLCONV RUN ${RUN}   DATA ${DATACFG}   N_FILTERS ${N_FILTERS}   DROPOUT ${DROPOUT}"
        if [ "$DRY_RUN" == "run" ]; then
            python train.py \
                --model AllConvTags --data ${DATASETS[${DATACFG}]} \
                --model_params "{l2: 0.0, n_filters: ${N_FILTERS}, dropout: ${DROPOUT}, tags_weight_filters: true}" \
                --out ${OUT}
        fi
    fi

    OUT=allconv_${DATACFG}_${N_FILTERS}_${DROPOUT}_tagsatsoftmax_${RUN}
    if ! [ -d "/home/filip/experiments/keyest/${OUT}" ]; then
        echo "========== ALLCONV RUN ${RUN}   DATA ${DATACFG}   N_FILTERS ${N_FILTERS}   DROPOUT ${DROPOUT}"
        if [ "$DRY_RUN" == "run" ]; then
            python train.py \
                --model AllConvTags --data ${DATASETS[${DATACFG}]} \
                --model_params "{l2: 0.0, n_filters: ${N_FILTERS}, dropout: ${DROPOUT}, tags_at_softmax: true}" \
                --out ${OUT}
        fi
    fi
done
