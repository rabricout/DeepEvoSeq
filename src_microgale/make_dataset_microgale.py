import random
import pickle
import os

from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO



data_path_a1 = Path('DATA_mmseqs/DATA_MICROGALE_clustered/A1_clustered')
species = ['MICROGALE', 'CHRYSOCHLORIS', 'ELEPHANTULUS', 'ORYCTEROPUS', 'PROCAVIA', 'TRUCHECHUS']
data_paths = {s: Path('DATA_mmseqs/DATA_MICROGALE_clustered/'+s) for s in species}


print('> loading raw microgale data')
all_data = {}
for s in species:
    data_raw = {}
    files = [f.name for f in data_paths[s].iterdir() if f.is_file()]
    for f in tqdm(files):
        for i, record in enumerate(SeqIO.parse(data_paths[s]/f, "fasta")):
            #s_id = record.id.split('Aplodontia_')[-1]
            s_id = f.split('.fasta')[0]
            data_raw[s_id] = str(record.seq)
    all_data[s] = data_raw



print('> loading clustered a1 data')
data = {k:{} for k in species}
data['A1'] = {}
del data['MICROGALE']
# data_a1 = {}
data_microgale = {}
clusters = {}
files = [f.name for f in data_path_a1.iterdir() if f.is_file()]
for k, f in tqdm(enumerate(files)):
    for i, record in enumerate(SeqIO.parse(data_path_a1/f, "fasta")):
        s_id = record.id.split('ancestor_')[-1]
        if k not in clusters:
            clusters[k] = []
        clusters[k].append(s_id)
        # data_a1[k] = (s_id, str(record.seq))
        data_microgale[s_id] = (s_id, all_data['MICROGALE'][s_id])
        data['A1'][s_id] = ('A1'+'_'+s_id, str(record.seq))
        for s in species:
            if s != 'MICROGALE':
                data[s][s_id] = (s+'_'+s_id, all_data[s][s_id])



print('> shuffling and splitting into train and eval subsets')
clusters_ids = list(clusters.keys())
random.shuffle(clusters_ids)
split_idx = int(0.8*len(clusters_ids))
train_clusters = clusters_ids[:split_idx]
eval_clusters = clusters_ids[split_idx:]

train_all = []
train_microgale = []
for c in train_clusters:
    for m_id in clusters[c]:
        train_all.append([data[s][m_id] for s in data.keys()])
        train_microgale.append(data_microgale[m_id])

eval_all = []
eval_microgale = []
for c in eval_clusters:
    for m_id in clusters[c]:
        eval_all.append([data[s][m_id] for s in data.keys()])
        eval_microgale.append(data_microgale[m_id])



print('> saving split data')
m_dir = 'DATA/DATA_MICROGALE/'
os.makedirs(m_dir, exist_ok=True)
with open(m_dir+"train_all.pkl", "wb") as f:
    pickle.dump(train_all, f)
with open(m_dir+"eval_all.pkl", "wb") as f:
    pickle.dump(eval_all, f)
with open(m_dir+"train_microgale.pkl", "wb") as f:
    pickle.dump(train_microgale, f)
with open(m_dir+"eval_microgale.pkl", "wb") as f:
    pickle.dump(eval_microgale, f)
