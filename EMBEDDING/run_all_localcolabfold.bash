#!/bin/bash

mkdir -p A1_SINGLE_REPR

for f in FASTA_DATA/DATA_A1/* ; do
    filename="${f##*/}"
    echo "$filename"
    if [ -d "A1_SINGLE_REPR/${filename}.output" ]; then
        bash localcolabfold/run_colabfoldbatch_sample.sh "$f" "A1_SINGLE_REPR/${filename}.output"
    fi
done
