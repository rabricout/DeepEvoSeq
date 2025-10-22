#!/bin/bash

nb_gpu=$1
gpu=$2
nature_only=$3
id=$4
nb_iter=$5
hyper_config=$6
strat=$7

for i in $(seq 0 $nb_iter)
do
    #bash _run_command.bash $nb_gpu $gpu PRECOMPUTED_DATASET/FINAL/ $nature_only $id $hyper_config
    bash _run_command.bash $nb_gpu $gpu PRECOMPUTED_DATASET_MUSCARDINUS/FINAL/ $nature_only $id $hyper_config $strat
done
