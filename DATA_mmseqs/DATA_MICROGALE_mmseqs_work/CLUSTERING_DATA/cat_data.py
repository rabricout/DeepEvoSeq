from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm


a1_dir = Path('../A1')
aplo_dir = Path('../MICROGALE')

# Cat a1 data
files = [f.name for f in a1_dir.iterdir() if f.is_file()]
records = []
for f in tqdm(files):
    for i, record in enumerate(SeqIO.parse(a1_dir/f, "fasta")):
        record.id += '_'+f.split('.fasta')[0]
        records.append(record)

SeqIO.write(records, 'a1.fa', 'fasta')

# Cat Microgale data
files = [f.name for f in aplo_dir.iterdir() if f.is_file()]
records = []
for f in tqdm(files):
    for i, record in enumerate(SeqIO.parse(aplo_dir/f, "fasta")):
        record.id += '_'+f.split('.fasta')[0]
        records.append(record)

SeqIO.write(records, 'microgale.fa', 'fasta')