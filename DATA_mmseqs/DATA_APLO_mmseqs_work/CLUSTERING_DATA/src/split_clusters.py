from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm

import os

file = 'clusterRes_all_seqs.fasta'
out_d = 'CLUSTERED_DATA/'
os.makedirs(out_d, exist_ok=True)

records = []
c = 0
for i, record in tqdm(enumerate(SeqIO.parse(file, "fasta"))):
    if record.seq == '':
        if len(records) > 0:
            SeqIO.write(records, out_d+str(c)+'.fa', 'fasta')
            records = []
            c += 1
    else:
        records.append(record)
SeqIO.write(records, out_d+str(c)+'.fa', 'fasta')