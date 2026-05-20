mkdir -p DATA_ALIGNED/
for fasta in DATA_ALL_A1/*.fasta; do
  mafft --auto "$fasta" > "DATA_ALIGNED/$(basename "$fasta" .fasta).fasta"
done