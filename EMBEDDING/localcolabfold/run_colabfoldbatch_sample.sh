#!/bin/bash

INPUTFILE="$1"
OUTPUTDIR="$2"
RANDOMSEED=0

export PATH="/home/localcolabfold/.pixi/envs/default/bin:${PATH}"

colabfold_batch \
  --num-recycle 3 \
  --amber \
  --templates \
  --use-gpu-relax \
  --num-models 2 \
  --model-order 1,2 \
  --random-seed ${RANDOMSEED} \
  --save-single-representations \
  ${INPUTFILE} \
  ${OUTPUTDIR}
