import random
import pickle
import os

from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO



data_path_a1 = Path('DATA_mmseqs/DATA_MICROGALE_clustered/A1_clustered')
data_path_microgale = Path('DATA_mmseqs/DATA_MICROGALE_clustered/MICROGALE')
data_path_chryso = Path('DATA_mmseqs/DATA_MICROGALE_clustered/CHRYSO')



print('> loading raw microgale data')
data_microgale_raw = {}
files = [f.name for f in data_path_microgale.iterdir() if f.is_file()]
for f in tqdm(files):
    for i, record in enumerate(SeqIO.parse(data_path_microgale/f, "fasta")):
        s_id = record.id.split('Microgale_')[-1]
        data_microgale_raw[s_id] = str(record.seq)



print('> loading raw chryso data')
data_chryso_raw = {}
files = [f.name for f in data_path_chryso.iterdir() if f.is_file()]
for f in tqdm(files):
    for i, record in enumerate(SeqIO.parse(data_path_chryso/f, "fasta")):
        s_id = f.split('.fasta')[0]
        data_chryso_raw[s_id] = str(record.seq)



print('> loading clustered a1 data')
data_a1 = {}
data_microgale = {}
data_chryso = {}
clusters = {}
files = [f.name for f in data_path_a1.iterdir() if f.is_file()]
for k, f in tqdm(enumerate(files)):
    for i, record in enumerate(SeqIO.parse(data_path_a1/f, "fasta")):
        s_id = record.id.split('ancestor_')[-1]
        if k not in clusters:
            clusters[k] = []
        clusters[k].append(s_id)
        data_a1[k] = (s_id, str(record.seq))
        data_microgale[k] = (s_id, data_microgale_raw[s_id])
        data_chryso[k] = (s_id, data_chryso_raw[s_id])



print('> shuffling and splitting into train and eval subsets')
clusters_ids = list(clusters.keys())
random.shuffle(clusters_ids)
split_idx = int(0.8*len(clusters_ids))
train_clusters = clusters_ids[:split_idx]
eval_clusters = clusters_ids[split_idx:]

train_microgale = [v for k, v in data_microgale.items() if k in train_clusters]
eval_microgale  = [v for k, v in data_microgale.items() if k in eval_clusters]
train_chryso = [v for k, v in data_chryso.items() if k in train_clusters]
eval_chryso  = [v for k, v in data_chryso.items() if k in eval_clusters]
train_a1 = [v for k, v in data_a1.items() if k in train_clusters]
eval_a1  = [v for k, v in data_a1.items() if k in eval_clusters]



print('> saving split data')
m_dir = 'DATA/DATA_MICROGALE/'
os.makedirs(m_dir, exist_ok=True)
with open(m_dir+"train_microgale.pkl", "wb") as f:
    pickle.dump(train_microgale, f)
with open(m_dir+"eval_microgale.pkl", "wb") as f:
    pickle.dump(eval_microgale, f)
with open(m_dir+"train_chryso.pkl", "wb") as f:
    pickle.dump(train_chryso, f)
with open(m_dir+"eval_chryso.pkl", "wb") as f:
    pickle.dump(eval_chryso, f)
with open(m_dir+"train_a1.pkl", "wb") as f:
    pickle.dump(train_a1, f)
with open(m_dir+"eval_a1.pkl", "wb") as f:
    pickle.dump(eval_a1, f)