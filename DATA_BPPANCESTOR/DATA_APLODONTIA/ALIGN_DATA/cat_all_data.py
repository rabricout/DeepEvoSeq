from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm

import os

aplo = Path('DATA_ALIGN_APLODONTIA_HUMAN/ALI_APLODONTIA')
human = Path('DATA_ALIGN_APLODONTIA_HUMAN/ALI_TARGET')
mouse = Path('DATA_ALIGN_APLODONTIA_MOUSE/ALI_TARGET')
musca = Path('DATA_ALIGN_APLODONTIA_MUSCARDINUS/ALI_TARGET')
sciurus = Path('DATA_ALIGN_APLODONTIA_SCIURUS/ALI_TARGET')

output = Path('DATA_CONCATENATED')
os.makedirs(output, exist_ok=True)

files = [f.name for f in aplo.iterdir() if f.is_file()]

for f in tqdm(files):
    cat_fasta = []
    try:
        for i, record in enumerate(SeqIO.parse(aplo/f, "fasta")):
            record.id = f"Aplodontia_{record.id}"
            record.seq = record.seq.replace("X", "")
            cat_fasta.append(record)
        for i, record in enumerate(SeqIO.parse(human/f, "fasta")):
            record.id = f"Homo_{record.id}"
            record.seq = record.seq.replace("X", "")
            cat_fasta.append(record)
        for i, record in enumerate(SeqIO.parse(mouse/f, "fasta")):
            record.id = f"Mouse_{record.id}"
            record.seq = record.seq.replace("X", "")
            cat_fasta.append(record)
        for i, record in enumerate(SeqIO.parse(musca/f, "fasta")):
            record.id = f"Muscardinus_{record.id}"
            record.seq = record.seq.replace("X", "")
            cat_fasta.append(record)
        for i, record in enumerate(SeqIO.parse(sciurus/f, "fasta")):
            record.id = f"Sciurus_{record.id}"
            record.seq = record.seq.replace("X", "")
            cat_fasta.append(record)
        with open(output/f, "w") as out:
            SeqIO.write(cat_fasta, out, "fasta")
    except Exception as e:
        #print('No complete alignment for', f)
        print(e)
        continue
