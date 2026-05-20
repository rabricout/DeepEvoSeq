from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm

import os

path_JC69 = Path('../BPPANCESTOR_JC69/DATA_FINAL_A1/')
path_LG08 = Path('../BPPANCESTOR_LG08/DATA_FINAL_A1/')
path_LG08_F = Path('../BPPANCESTOR_LG08_F/DATA_FINAL_A1/')
path_WAG01_F = Path('../BPPANCESTOR_WAG01_F/DATA_FINAL_A1/')

output = Path('DATA_ALL_A1')
os.makedirs(output, exist_ok=True)

files = [f.name for f in path_JC69.iterdir() if f.is_file()]

for f in tqdm(files):
    cat_fasta = []
    try:
        for i, record in enumerate(SeqIO.parse(path_JC69/f, "fasta")):
            record.id = f"JC69_{record.id}"
            cat_fasta.append(record)
        for i, record in enumerate(SeqIO.parse(path_LG08/f, "fasta")):
            record.id = f"LG08_{record.id}"
            cat_fasta.append(record)
        for i, record in enumerate(SeqIO.parse(path_LG08_F/f, "fasta")):
            record.id = f"LG08+F_{record.id}"
            cat_fasta.append(record)
        for i, record in enumerate(SeqIO.parse(path_WAG01_F/f, "fasta")):
            record.id = f"WAG01+F_{record.id}"
            cat_fasta.append(record)
        with open(output/f, "w") as out:
            SeqIO.write(cat_fasta, out, "fasta")
    except Exception as e:
        #print('No complete alignment for', f)
        #print(e)
        continue