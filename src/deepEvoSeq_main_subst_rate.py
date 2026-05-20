import torch.nn as nn
import numpy as np
import argparse
import json
import esm

from torch.utils.tensorboard import SummaryWriter
from deepEvoSeq_models_subst_rate import *
from torch.utils.data import DataLoader
from deepEvoSeq_dataset import *
from utils import *
from tqdm import tqdm
from datetime import datetime



# Parsing
parser = argparse.ArgumentParser(description="DeepEvoSeq options")
parser.add_argument("-r", "--lr", type=float, help="Learning rate", default=1e-3)
parser.add_argument("-s", "--simple", action="store_true", help="Without ESM embedding")
parser.add_argument("-a", "--attention", action="store_true", help="Use trainable attention layers")

args = parser.parse_args()
lr = args.lr
is_simple = args.simple
use_attention = args.attention



# Load a pretrained ESM model and alphabet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('> using', device)
model_esm_body, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model_esm_body = model_esm_body.to(device)
model_esm_body.eval()  # disables dropout
batch_converter = alphabet.get_batch_converter()



# Defining dict to go from aa to idx
unique_labels = [a for a in '-ARNDCEQGHILKMSPFTWYV']
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}



# Prepare datasets
data_path = 'DATA/DATA_APLO/'
print('> constructing datasets and dataloaders')
BATCH_SIZE=32
NUM_WORKERS=4
if is_simple:
    train_dataset = DeepEvoSeqSimpleDatasetNatureSubstRate(label_to_idx, data_path+'train_all.pkl', data_path+'train_aplo.pkl')
    eval_dataset = DeepEvoSeqSimpleDatasetNatureSubstRate(label_to_idx, data_path+'eval_all.pkl', data_path+'eval_aplo.pkl')
else:
    train_dataset = DeepEvoSeqDatasetNatureSubstRate(label_to_idx, data_path+'train_all.pkl', data_path+'train_aplo.pkl')
    eval_dataset = DeepEvoSeqDatasetNatureSubstRate(label_to_idx, data_path+'eval_all.pkl', data_path+'eval_aplo.pkl')
nb_amino_acids = 21
nb_species = len(train_dataset[0][0][0])


model_deepEvoSeq = DeepEvoSeqGeneric(nb_species, nb_amino_acids, attention=use_attention, is_simple=is_simple)
model_deepEvoSeq = model_deepEvoSeq.to(device)
if is_simple:
    collate_fn = collate_fn_aa_subst_rate_simple
else:
    collate_fn = collate_fn_aa_subst_rate


# Dataset
# class BaselineDataset(Dataset):
#     def __init__(self, data_path: str, labels_path: str):
#         with open(data_path, "rb") as f:
#             self.data = pickle.load(f)
#         with open(labels_path, "rb") as f:
#             self.labels = pickle.load(f)

#         assert len(self.data) == len(self.labels), "Data and labels must have same length"
#         data_squirrel = []
#         data_musca = []
#         data_a1 = []
#         for i, d in enumerate(self.data):
#             for m_id, s in d:
#                 if 'A1_' in m_id:
#                     data_a1.append(s)
#         subst_rate = []
#         for i, yi in enumerate(self.labels):
#             seq_target = self.labels[i][1]
#             seq_a1 = data_a1[i]
#             p = [seq_target[k]!=seq_a1[k] for k in range(len(seq_target))]
#             subst_rate.append(sum(p)/len(p))
#         self.data_musca = data_musca
#         self.data_squirrel = data_squirrel
#         self.labels = subst_rate        

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         x = self.data[idx]   # e.g., a tensor or numpy array of shape (L, D)
#         y = self.labels[idx] # e.g., int or tensor of shape (L,) or (L,)
#         return x, y



# Prepare datasets
# data_path = 'DATA/DATA_APLO/'
# print('> constructing datasets and dataloaders')
# train_dataset = BaselineDataset(data_path+'train_all.pkl', data_path+'train_aplo.pkl')
# # rate_target_squirrel, rate_musca_squirrel = train_dataset.mean_rates()
# eval_dataset = BaselineDataset(data_path+'eval_all.pkl', data_path+'eval_aplo.pkl')
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn = collate_fn,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,  # if using GPU
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=1,
    collate_fn = collate_fn,
    shuffle=False,
    pin_memory=False,  # if using GPU
)



# Training loop
optimizer = torch.optim.Adam(model_deepEvoSeq.parameters(), lr=lr)
num_epochs = 500
date_str = datetime.now().strftime("%m_%d_%Y_%H_%M")
run_id = ('simple_' if is_simple else 'full_') + str(lr) + '_'
writer = SummaryWriter("runs_subst_rate/"+run_id+date_str)
with open("runs_subst_rate/specs_"+run_id+date_str+'.txt', 'w', encoding='utf-8') as f:    # save model specs and training hyperparameters
    model_specs = model_deepEvoSeq.give_specs_dict()
    training_specs = {'lr':lr, 'batch_size':BATCH_SIZE, 'num_workers': NUM_WORKERS}
    json.dump([model_specs, training_specs], f, indent=2)

for epoch in range(num_epochs):
    model_deepEvoSeq.train()
    for (batch_x, batch_x_rates), batch_y in tqdm(train_loader):
        batch_y = batch_y.to(device)
        if is_simple:
            batch_x = batch_x.to(device)
            batch_x_rates = batch_x_rates.to(device)
            pred_subst_rate = model_deepEvoSeq(batch_x, batch_x_rates)
        else:
            batch_x_swap = list(zip(*batch_x))
            reps_all = []
            for batch_species in batch_x_swap:
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_species)
                with torch.no_grad():
                    batch_tokens = batch_tokens.to(device)
                    esm_embedding = model_esm_body(batch_tokens, repr_layers=[6], return_contacts=False)
                reps = esm_embedding["representations"][6][:,1:-1:]  # layer 6 for this model
                reps_all.append(reps)
            reps_all = torch.stack(reps_all, dim=1).to(device)
            pred_subst_rate = model_deepEvoSeq(reps_all, batch_x_rates)

        criterion = nn.MSELoss()
        loss = criterion(pred_subst_rate, batch_y)
        diff = (pred_subst_rate-batch_y).abs().mean()
        writer.add_scalar("Loss/train", loss.cpu(), epoch)
        writer.add_scalar("Mean difference/train", diff.cpu(), epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model_deepEvoSeq.eval()
    diffs = []
    losses = []
    with torch.no_grad():
        for (batch_x, batch_x_rates), batch_y in tqdm(eval_loader):
            batch_y = batch_y.to(device)
            if is_simple:
                batch_x = batch_x.to(device)
                batch_x_rates = batch_x_rates.to(device)
                pred_subst_rate = model_deepEvoSeq(batch_x, batch_x_rates)
            else:
                batch_x_swap = list(zip(*batch_x))
                reps_all = []
                for batch_species in batch_x_swap:
                    batch_labels, batch_strs, batch_tokens = batch_converter(batch_species)
                    with torch.no_grad():
                        batch_tokens = batch_tokens.to(device)
                        esm_embedding = model_esm_body(batch_tokens, repr_layers=[6], return_contacts=False)
                    reps = esm_embedding["representations"][6][:,1:-1:]  # layer 6 for this model
                    reps_all.append(reps)
                reps_all = torch.stack(reps_all, dim=1).to(device)
                pred_subst_rate = model_deepEvoSeq(reps_all, batch_x_rates)

            criterion = nn.MSELoss()
            losses.append(criterion(pred_subst_rate, batch_y).cpu())
            diffs.append((pred_subst_rate-batch_y).abs().mean().cpu())
    print('Loss:', np.mean(np.array(losses)))
    print('Mean diff:', np.mean(np.array(diffs)))
