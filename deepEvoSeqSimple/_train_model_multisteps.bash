#!/bin/bash

nb_gpu=$1
gpu=$2
id=$3
nb_iter=$4
hyper_config=$5

for i in $(seq 0 $nb_iter)
do
    bash _run_command.bash $nb_gpu $gpu PRECOMPUTED_DATASET_MUSCARDINUS/FINAL/ $id $hyper_config
done
