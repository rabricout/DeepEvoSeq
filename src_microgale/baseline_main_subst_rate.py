import numpy as np
import argparse
import sys

from baseline_models_subst_rate import *
from torch.utils.data import DataLoader
from deepEvoSeq_dataset import *
from utils import *
from tqdm import tqdm
from datetime import datetime


# Parsing
parser = argparse.ArgumentParser(description="DeepEvoSeq options")
parser.add_argument("-m", "--model", type=str, help="Baseline model", default="Blosum")

args = parser.parse_args()
model_id = args.model
model = BaselineSubstRate()



class BaselineDataset(Dataset):
    def __init__(self, data_path: str, labels_path: str):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)

        assert len(self.data) == len(self.labels), "Data and labels must have same length"
        data_squirrel = []
        data_musca = []
        data_a1 = []
        for i, d in enumerate(self.data):
            for m_id, s in d:
                if 'MUSCA_' in m_id:
                    data_musca.append(s)
                if 'SCIURUS_' in m_id:
                    data_squirrel.append(s)
                if 'A1_' in m_id:
                    data_a1.append(s)
        subst_rate = []
        rates_musca_squirrel = []
        rates_target_squirrel = []
        for i, yi in enumerate(self.labels):
            seq_target = self.labels[i][1]
            seq_a1 = data_a1[i]
            seq_squirrel = data_squirrel[i]
            seq_musca = data_musca[i]
            p = [seq_target[k]!=seq_a1[k] for k in range(len(seq_target))]
            subst_rate.append(sum(p)/len(p))
            p = [seq_target[k]!=seq_squirrel[k] for k in range(len(seq_target))]
            rates_target_squirrel.append(sum(p)/len(p))
            p = [seq_musca[k]!=seq_squirrel[k] for k in range(len(seq_musca))]
            rates_musca_squirrel.append(sum(p)/len(p))

        self.rate_musca_squirrel = sum(rates_musca_squirrel) / len(rates_musca_squirrel)
        self.rate_target_squirrel = sum(rates_target_squirrel) / len(rates_target_squirrel)
        self.data_musca = data_musca
        self.data_squirrel = data_squirrel
        self.labels = subst_rate        

    def mean_rates(self):
        return self.rate_target_squirrel, self.rate_musca_squirrel

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_musca = self.data_musca[idx]   # e.g., a tensor or numpy array of shape (L, D)
        x_squirrel = self.data_squirrel[idx]   # e.g., a tensor or numpy array of shape (L, D)
        y = self.labels[idx] # e.g., int or tensor of shape (L,) or (L,)
        return (x_musca, x_squirrel), y



# Prepare datasets
data_path = 'DATA/DATA_APLO/'
print('> constructing datasets and dataloaders')
train_dataset = BaselineDataset(data_path+'train_all.pkl', data_path+'train_aplo.pkl')
rate_target_squirrel, rate_musca_squirrel = train_dataset.mean_rates()
eval_dataset = BaselineDataset(data_path+'eval_all.pkl', data_path+'eval_aplo.pkl')
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=1,
#     shuffle=True,
#     num_workers=1,
#     pin_memory=False,  # if using GPU
# )
eval_loader = DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=False,  # if using GPU
)


date_str = datetime.now().strftime("%m_%d_%Y_%H_%M")
run_id = model_id
num_epochs=1

for epoch in range(num_epochs):
    diffs = []
    for (batch_x_homo, batch_x_squirrel), batch_y in tqdm(eval_loader):
        pred_subst_rate = np.array(model.forward(batch_x_homo[0], batch_x_squirrel[0]))
        pred_subst_rate = pred_subst_rate / rate_musca_squirrel * rate_target_squirrel / 2
        diffs.append(pred_subst_rate - batch_y[0].item())
    print('Mean diff in subst rate between truth and predicted value using baseline:', np.mean(np.array(diffs)))
