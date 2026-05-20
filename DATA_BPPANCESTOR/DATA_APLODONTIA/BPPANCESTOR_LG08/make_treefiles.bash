mkdir -p DATA_TREEFILES/
cp -r DATA_ALIGNED tmp
cd tmp
for fasta in *.fasta; do
    iqtree2 -s $fasta -m MFP -B 1000
done
mv *.treefile ../DATA_TREEFILES
cd ..
rm -r tmp