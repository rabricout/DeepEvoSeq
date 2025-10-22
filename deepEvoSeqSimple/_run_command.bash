#!/bin/bash

nb_gpu=$1
gpu=$2
data_dir=$3
my_model_id=$4
hyper_config=$5
#checkpoint=$6


python3 train_openfold.py /databases/alphafold/pdb_mmcif/mmcif_files/ RESULTS/ $data_dir/TRAIN/ $data_dir/EVAL/ \
	2023-10-10 \
    --hyper_config $hyper_config \
    --train_epoch_len 2000 \
    --eval_epoch_len 400 \
    --config_preset my_model \
    --nb_gpus $nb_gpu \
    --gpu $gpu \
    --precision 32 \
    --resume_model_weights_only True \
    --log_performance False \
    --checkpoint_every_epoch \
    --my_model_id $my_model_id \
    --deepspeed_config_path JSON_FILES/deepspeed_config.json \

    exit 1


if [ -z "$checkpoint" ];
then (
python3 train_openfold.py /databases/alphafold/pdb_mmcif/mmcif_files/ RESULTS/ $3/TRAIN $3/EVAL \
    2023-10-10 \
    --nature_only $nature_only \
    --precision bf16 \
    --train_epoch_len 5000 \
    --eval_epoch_len 1000 \
    --config_preset my_model \
    --gpus $nb_gpu \
    --gpu $gpu \
    --seed $seed \
    --resume_model_weights_only True \
    --deepspeed_config_path JSON_FILES/deepspeed_config.json \
    --log_performance False \
    --max_epochs 1 \
    --my_model_id $my_model_id \
    )
else (
python3 train_openfold.py /databases/alphafold/pdb_mmcif/mmcif_files/ RESULTS/ $3/TRAIN $3/EVAL \
    2023-10-10 \
    --nature_only $nature_only \
    --precision bf16 \
    --train_epoch_len 50 \
    --eval_epoch_len 10 \
    --config_preset my_model \
    --gpus $nb_gpu \
    --gpu $gpu \
    --seed $seed \
    --resume_model_weights_only True \
    --deepspeed_config_path JSON_FILES/deepspeed_config.json \
    --log_performance False \
    --resume_from_ckpt $checkpoint \
    --checkpoint_every_epoch \
    --max_epochs 1 \
    --my_model_id $my_model_id \
    )
fi

exit 1

#$2/train $2/eval \
#--gpus 1 \

    --checkpoint_every_epoch \

    --resume_from_jax_params $2/params/params_model_1.npz \
    --val_data_dir $2/FINAL_DATA/EVAL/INPUT_CIF/ \
    --val_alignment_dir $2/FINAL_DATA/EVAL/INPUT_PRECOMPUTED/ \
    --template_release_dates_cache_path $2/JSON_FILES/mmcif_cache.json \
    --train_chain_data_cache_path $2/JSON_FILES/chain_data_cache.json \
    --eval_chain_data_cache_path $2/JSON_FILES/chain_data_cache_eval.json \
