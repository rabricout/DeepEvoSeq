import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os

from baseline_models_position import *
from torch.utils.data import DataLoader
from deepEvoSeq_dataset import *
from utils import *
from tqdm import tqdm
from datetime import datetime


# Parsing
parser = argparse.ArgumentParser(description="DeepEvoSeq options")
parser.add_argument("-m", "--model", type=str, help="Baseline model", default="Random")

args = parser.parse_args()
model_id = args.model
if model_id=='Random':
    model = BaselineRandom()
elif model_id=='Consensus':
    model = BaselineConsensus()
elif model_id=='Proxy':
    model = BaselineProxy()
elif model_id=='Species':
    model = BaselineSpecies()
else:
    print('Model not recognized')
    sys.exit()



class BaselineDataset(Dataset):
    def __init__(self, data_path: str, labels_path: str):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)

        assert len(self.data) == len(self.labels), "Data and labels must have same length"
        data_a1 = []
        data = []
        for i, d in enumerate(self.data):
            datum = []
            sorted_d = dict(sorted(dict(d).items()))
            ids_raw = list(sorted_d.keys())
            self.ids = [m_id.split('_')[0] for m_id in ids_raw]

            for m_id, s in sorted_d.items():
                datum.append(s)
                if 'A1_' in m_id:
                    data_a1.append(s)
            data.append(datum)
        positions = []
        # Building substitution mask for the position task
        for i, yi in enumerate(self.labels):
            seq_target = self.labels[i][1]
            seq_a1 = data_a1[i]
            p = [seq_target[k]!=seq_a1[k] for k in range(len(seq_target))]
            positions.append(p)
        self.positions = positions
        self.data = data

    def get_specices_ids(self):
        return self.ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        p = np.array(self.positions[idx])
        return x, p



# Prepare datasets
data_path = 'DATA/DATA_MICROGALE/'
print('> constructing datasets and dataloaders')
train_dataset = BaselineDataset(data_path+'train_all.pkl', data_path+'train_microgale.pkl')
eval_dataset = BaselineDataset(data_path+'eval_all.pkl', data_path+'eval_microgale.pkl')
ids = train_dataset.get_specices_ids()
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=False,  # if using GPU
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=False,  # if using GPU
)
print('> species:', ids)
model.train(train_loader)
model.set_species(ids, 'CHRYSOCHLORIS')    # only useful for "Species" baseline

date_str = datetime.now().strftime("%m_%d_%Y_%H_%M")
run_id = model_id
num_epochs=1

data_to_save = []
for epoch in range(num_epochs):
    accs = []
    pos = 0
    tot = 0
    for batch_x, batch_p in tqdm(eval_loader):
        subst_pos = np.array(model.forward(batch_x))
        batch_p = np.array(batch_p[0])
        matchs = (subst_pos == batch_p)
        if len(matchs) > 0:
            acc = np.sum(matchs)/len(matchs)
            pos += np.sum(matchs)
            tot += len(matchs)
            accs.append(acc)
        
        data_to_save.append({'labels': np.array(batch_p).astype(int), 'preds': subst_pos})
    print(np.mean(np.array(accs)))
    print(pos/tot)

os.makedirs('runs_position_microgale/'+model_id, exist_ok=True)
np.save('runs_position_microgale/'+model_id+'/position_eval_values.npy', [data_to_save])