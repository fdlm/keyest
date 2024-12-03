#!/usr/bin/env bash
export THEANO_FLAGS=device=cuda2

for RUN in {1..10}; do
#    for N_FILTERS in 8 16 24; do
    for N_FILTERS in 16 24; do
#        python train.py --model Eusipco2017Snippet --model_params "{n_filters: ${N_FILTERS}, l2: 0}"  --data giantsteps --out gs_${N_FILTERS}_${RUN}
#        python train.py --model Eusipco2017Snippet --model_params "{n_filters: ${N_FILTERS}, l2: 0}"  --data cmdb --out cm_${N_FILTERS}_${RUN}
        python train.py --model Eusipco2017Snippet --model_params "{n_filters: ${N_FILTERS}, l2: 0}"  --data billboard --out bb_${N_FILTERS}_${RUN}
    done

#    for N_FILTERS in 8 16 24; do
    for N_FILTERS in 24; do
#        python train.py --model Eusipco2017Snippet --model_params "{n_filters: ${N_FILTERS}, l2: 0}" --data giantsteps,cmdb --out gscm_${N_FILTERS}_${RUN}
#        python train.py --model Eusipco2017Snippet --model_params "{n_filters: ${N_FILTERS}, l2: 0}" --data giantsteps,billboard --out gsbb_${N_FILTERS}_${RUN}
        python train.py --model Eusipco2017Snippet --model_params "{n_filters: ${N_FILTERS}, l2: 0}" --data cmdb,billboard --out cmbb_${N_FILTERS}_${RUN}
    done

#    for N_FILTERS in 8 16 24; do
#        python train.py --model Eusipco2017Snippet --model_params "{n_filters: ${N_FILTERS}, l2: 0}"  --data giantsteps,cmdb,billboard --out gscmbb_${N_FILTERS}_${RUN}
#    done
done

