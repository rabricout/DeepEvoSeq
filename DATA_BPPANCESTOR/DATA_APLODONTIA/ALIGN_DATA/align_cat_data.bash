mkdir -p DATA_ALIGNED/
for fasta in DATA_CONCATENATED/*.fasta; do
  mafft --auto "$fasta" > "DATA_ALIGNED/$(basename "$fasta" .fasta).fasta"
done