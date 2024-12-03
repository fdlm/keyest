#!/usr/bin/env bash

export THEANO_FLAGS="device=${1:-cuda0}"
DRY_RUN=${2:-run}

if [ $# -gt 2 ]; then
    RUNS=${@:3}
else
    RUNS="0 1 2"
fi

declare -A DATASETS
DATASETS[gs]=giantsteps
DATASETS[bb]=billboard
DATASETS[cm]=cmdb
DATASETS[gsbb]=${DATASETS[gs]},${DATASETS[bb]}
DATASETS[gscm]=${DATASETS[gs]},${DATASETS[cm]}
DATASETS[cmbb]=${DATASETS[cm]},${DATASETS[bb]}
DATASETS[gscmbb]=${DATASETS[gs]},${DATASETS[cm]},${DATASETS[bb]}

DATACFG=gscmbb
DROPOUT=0.0
for RUN in ${RUNS}; do
    for N_FILTERS in 4 8 12; do
        for TEACHER_TEMP in 1.0 3.0 6.0 12.0; do
            for TEACHER_FACT in 1.0 2.0 4.0; do
                for GT_FACT in 0.0 0.5 1.0; do
                    OUT=distilled_allconv_${N_FILTERS}_TT=${TEACHER_TEMP}_TF=${TEACHER_FACT}_GF=${GT_FACT}_${RUN}
                    if ! [ -d "/home/filip/experiments/keyest/${OUT}" ]; then
                        echo "========== ALLCONV RUN ${RUN}  N_FILTERS ${N_FILTERS} T ${TEACHER_TEMP} TF ${TEACHER_FACT} GF ${GT_FACT}"
                        if [ "$DRY_RUN" == "run" ]; then
                            python train.py \
                                --model AllConvDistilled --data ${DATASETS[${DATACFG}]} \
                                --model_params "{l2: 0.0, n_filters: ${N_FILTERS}, dropout: ${DROPOUT}, \
                                                 temperature: ${TEACHER_TEMP}, teacher_factor: ${TEACHER_FACT}, \
                                                 gt_factor: ${GT_FACT}}" \
                                --out ${OUT}
                        fi
                    fi
                done
            done
        done
    done
done
