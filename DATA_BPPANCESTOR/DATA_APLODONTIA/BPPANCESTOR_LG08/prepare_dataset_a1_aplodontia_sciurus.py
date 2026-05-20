from Bio.SeqRecord import SeqRecord
from pathlib import Path
from Bio.Seq import Seq
from Bio import SeqIO
from tqdm import tqdm

import numpy as np

fasta_dir = Path('DATA_A1')

files = [f.name for f in fasta_dir.iterdir() if f.is_file()]
out_path = {}
for i, record in enumerate(SeqIO.parse(fasta_dir/files[0], "fasta")):
    m_id = record.id.split('_')[0]
    out_path[m_id] = Path('DATA_final_'+str(m_id))
    out_path[m_id].mkdir(exist_ok=True)

for f in tqdm(files):
    seqs = dict.fromkeys(out_path.keys())
    ids = dict.fromkeys(out_path.keys())
    for i, record in enumerate(SeqIO.parse(fasta_dir/f, "fasta")):
        m_id = record.id.split('_')[0]
        seqs[m_id] = record.seq
        ids[m_id] = record.id

    for k, v in seqs.items():
        if v == None:
            raise ValueError("One or several IDs not found in alignment")

    # Remove positions (columns) where at least one has a gap in aplo
    cols_to_keep = np.array([1 if a !='-' else 0 for a in seqs['Aplodontia']])
    for k, v in seqs.items():
        subsampled = ''.join(np.array(list(v))[cols_to_keep.astype(bool)])
        rec = SeqRecord(Seq(subsampled), id=ids[k], description="")
        SeqIO.write([rec], out_path[k]/f, "fasta")