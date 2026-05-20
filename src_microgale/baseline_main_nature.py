import numpy as np
import argparse
import pickle
import sys
import os

from baseline_models_nature import *
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
unique_labels = [a for a in '-ARNDCEQGHILKMSPFTWYV']
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}



class BaselineDataset(Dataset):
    def __init__(self, data_path: str, labels_path: str):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)

        assert len(self.data) == len(self.labels), "Data and labels must have same length"
        data_a1 = []
        data_species = []
        for i, d in enumerate(self.data):
            for m_id, s in d:
                if 'A1_' in m_id:
                    data_a1.append(s)
                if 'CHRYSOCHLORIS_' in m_id:
                    data_species.append(s)
        positions = []
        # Building substitution mask for the position task
        for i, yi in enumerate(self.labels):
            seq_target = self.labels[i][1]
            seq_a1 = data_a1[i]
            p = [seq_target[k]!=seq_a1[k] for k in range(len(seq_target))]
            positions.append(p)
        self.positions = positions
        self.data = data_species    # data_a1

    def save_transition_dict(self):
        transitionMatrix = np.zeros((len(unique_labels), len(unique_labels)))
        for i, d in enumerate(self.data):
            for k in range(len(d)):
                if self.data[i][k] != self.labels[i][1][k]:
                    transitionMatrix[label_to_idx[self.data[i][k]], label_to_idx[self.labels[i][1][k]]] += 1
        transition_matrix_norm = transitionMatrix / transitionMatrix.sum(axis=1, keepdims=True)
        transition_dict = {}
        for i, label in enumerate(unique_labels):
            transition_dict[label] = transition_matrix_norm[i]
        # transition_dict = {}
        # for i in range(len(unique_labels)):
        #     transition_dict[idx_to_label[i]] = idx_to_label[np.argmax(transitionMatrix[i])]
        with open('src/additional_data/transition_matrix_dict.pkl', 'wb') as f:
            pickle.dump(transition_dict, f)

    def subst_rate(self):
        diff, tot = 0, 0
        for i, a1_seq in enumerate(self.data):
            sim = [a1_seq[k] != self.labels[i][1][k] for k in range(len(a1_seq))]
            diff += sum(sim)
            tot += len(sim)
        return diff/tot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # e.g., a tensor or numpy array of shape (L, D)
        y = self.labels[idx][1] # e.g., int or tensor of shape (L,) or (L,)
        p = np.array(self.positions[idx])
        return (x, x), (y,p)
    


# Prepare datasets
data_path = 'DATA/DATA_MICROGALE/'
print('> constructing datasets and dataloaders')
train_dataset = BaselineDataset(data_path+'train_all.pkl', data_path+'train_microgale.pkl')
train_dataset.save_transition_dict()
eval_dataset = BaselineDataset(data_path+'eval_all.pkl', data_path+'eval_microgale.pkl')
eval_loader = DataLoader(
    eval_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=False,  # if using GPU
)



# Loading model
if model_id=='Blosum':
    model = BaselineBlosum(method='argmax')
elif model_id=='TransitionMatrix':
    model = BaselineTransitionMatrix(method='argmax')
elif model_id=='GeneticCode':
    model = BaselineGeneticCode()
else:
    sys.exit()



date_str = datetime.now().strftime("%m_%d_%Y_%H_%M")
run_id = model_id
num_epochs=1

data_to_save = []
for epoch in range(num_epochs):
    accs = []
    pos = 0
    tot = 0
    for (batch_x, batch_x_a1), (batch_y, batch_p) in tqdm(eval_loader):
        subst = np.array(model.forward(batch_x[0]))
        batch_y_masked = np.array(list(batch_y[0]))
        batch_p = batch_p[0]
        batch_y_masked[~batch_p] = ':'  # ":" is ignore_index
        matchs = (subst[batch_p] == batch_y_masked[batch_p])
        if len(matchs) > 0:
            acc = np.sum(matchs)/len(matchs)
            pos += np.sum(matchs)
            tot += len(matchs)
            accs.append(acc)
        
        subst_int = [label_to_idx[a] for a in subst]
        n_classes = len(unique_labels)
        one_hot = np.eye(n_classes)[subst_int]
        batch_y = [label_to_idx[a] for a in batch_y[0]]
        batch_x_a1_save = [label_to_idx[a] for a in batch_x_a1[0]]
        data_to_save.append({'labels': np.array([batch_y]), 'preds': np.array(one_hot), 'positions': np.array([batch_p]), 'a1': np.array([batch_x_a1_save])})

    print(np.mean(np.array(accs)))
    print(pos/tot)
os.makedirs('runs_nature_microgale/'+model_id, exist_ok=True)
np.save('runs_nature_microgale/'+model_id+'/nature_eval_values.npy', data_to_save)