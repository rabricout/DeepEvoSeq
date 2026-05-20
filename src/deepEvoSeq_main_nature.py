import numpy as np
import argparse
import torch
import json
import esm

import torch.nn.functional as F

from torch.utils.data import DataLoader
from deepEvoSeq_models_nature import *
from deepEvoSeq_dataset import *
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchinfo import summary

# Parsing
parser = argparse.ArgumentParser(description="DeepEvoSeq options")
parser.add_argument("-r", "--lr", type=float, help="Learning rate", default=5e-4)
parser.add_argument("-s", "--simple", action="store_true", help="Without ESM embedding")
parser.add_argument("-a", "--attention", action="store_true", help="Use trainable attention layers")
parser.add_argument("-w", "--window", type=int, help="Window size", default=-1)

args = parser.parse_args()
lr = args.lr
is_simple = args.simple
use_attention = args.attention
window = args.window


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
    train_dataset = DeepEvoSeqSimpleDatasetNature(label_to_idx, data_path+'train_all.pkl', data_path+'train_aplo.pkl', window=window)
    eval_dataset = DeepEvoSeqSimpleDatasetNature(label_to_idx, data_path+'eval_all.pkl', data_path+'eval_aplo.pkl')
else:
    train_dataset = DeepEvoSeqDatasetNature(label_to_idx, data_path+'train_all.pkl', data_path+'train_aplo.pkl', window=window)
    eval_dataset = DeepEvoSeqDatasetNature(label_to_idx, data_path+'eval_all.pkl', data_path+'eval_aplo.pkl')
nb_amino_acids = 21
nb_species = 4


# Prepare model based on attention and is_simple
model_deepEvoSeq = DeepEvoSeqGeneric(nb_species, nb_amino_acids, attention=use_attention, is_simple=is_simple, attn_heads=8)
model_deepEvoSeq = model_deepEvoSeq.to(device)
for name, module in model_deepEvoSeq.named_modules():
    print(name, "->", module)
if is_simple:
    collate_fn = collate_fn_aa_simple
else:
    collate_fn = collate_fn_aa

# Prepare dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn = collate_fn,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,  # if using GPU
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=1,
    collate_fn = collate_fn,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,  # if using GPU
)

# Training loop
optimizer = torch.optim.Adam(model_deepEvoSeq.parameters(), lr=lr)
num_epochs = 300 if is_simple else 150
date_str = datetime.now().strftime("%m_%d_%Y_%H_%M")
run_id = ('simple_' if is_simple else 'full_') + str(lr) + '_' + ('w_'+str(window)+'_' if window>0 else '')
writer = SummaryWriter("runs_nature/"+run_id+date_str)
with open("runs_nature/"+run_id+date_str+'/specs.txt', 'w', encoding='utf-8') as f:    # save model specs and training hyperparameters
    model_specs = model_deepEvoSeq.give_specs_dict()
    training_specs = {'lr':lr, 'batch_size':BATCH_SIZE, 'num_workers': NUM_WORKERS}
    json.dump([model_specs, training_specs], f, indent=2)

epoch_max_accs = {}
for epoch in range(num_epochs):
    # Training
    print('> epoch', epoch, '/', num_epochs)
    model_deepEvoSeq.train()
    for (batch_x, batch_a1), (batch_y, batch_p) in tqdm(train_loader):
        batch_y = batch_y.to(device)
        if is_simple:
            batch_x = batch_x.to(device)
            logits = model_deepEvoSeq(batch_x)
        else:
            batch_x_tensor, batch_x_raw = batch_x
            batch_x_swap = list(zip(*batch_x_raw))
            reps_all = []
            for batch_species in batch_x_swap:
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_species)
                with torch.no_grad():
                    batch_tokens = batch_tokens.to(device)
                    esm_embedding = model_esm_body(batch_tokens, repr_layers=[6], return_contacts=False)
                reps = esm_embedding["representations"][6][:,1:-1:]  # layer 6 for this model
                reps_all.append(reps)
            reps_all = torch.stack(reps_all, dim=1).to(device)
            batch_x_tensor = batch_x_tensor.to(device)
            logits = model_deepEvoSeq(batch_x_tensor, reps_all)

        batch_y_masked = batch_y.clone()
        batch_y_masked[~batch_p] = -100  # -100 is ignore_index
        loss = F.cross_entropy(
            logits.reshape(-1, nb_amino_acids),
            batch_y_masked.reshape(-1),
            ignore_index=-100,
        )
        acc = (logits[batch_p].argmax(dim=-1) == batch_y_masked[batch_p]).float().mean()
        writer.add_scalar("Loss/train", loss.cpu(), epoch)
        writer.add_scalar("Accuracy/train", acc.cpu(), epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model_deepEvoSeq.eval()
    accs = []
    losses = []
    data_epoch = []
    with torch.no_grad():
        for (batch_x, batch_x_a1), (batch_y, batch_p) in tqdm(eval_loader):
            batch_y = batch_y.to(device)
            if is_simple:
                batch_x = batch_x.to(device)
                logits = model_deepEvoSeq(batch_x)
            else:
                batch_x_tensor, batch_x_raw = batch_x
                batch_x_swap = list(zip(*batch_x_raw))
                reps_all = []
                for batch_species in batch_x_swap:
                    batch_labels, batch_strs, batch_tokens = batch_converter(batch_species)
                    with torch.no_grad():
                        batch_tokens = batch_tokens.to(device)
                        esm_embedding = model_esm_body(batch_tokens, repr_layers=[6], return_contacts=False)
                    reps = esm_embedding["representations"][6][:,1:-1:]  # layer 6 for this model
                    reps_all.append(reps)
                reps_all = torch.stack(reps_all, dim=1).to(device)
                batch_x_tensor = batch_x_tensor.to(device)
                logits = model_deepEvoSeq(batch_x_tensor, reps_all)

            batch_y_masked = batch_y.clone()
            batch_y_masked[~batch_p] = -100  # -100 is ignore_index
            loss = F.cross_entropy(
                logits.reshape(-1, nb_amino_acids),
                batch_y_masked.reshape(-1),
                ignore_index=-100,
            )
            if not torch.isnan(loss):
                losses.append(loss.cpu())
            acc = (logits[batch_p].argmax(dim=-1) == batch_y_masked[batch_p]).float()
            probs = F.softmax(logits[0], dim=1)
            data_epoch.append({'labels': np.array(batch_y.cpu()), 'preds': np.array(probs.cpu()), 'positions': np.array(batch_p.cpu()), 'a1': np.array(batch_x_a1.cpu())})
            if len(acc) > 0:
                acc = acc.mean()
                accs.append(acc.cpu())

    epoch_acc = np.mean(np.array(accs))
    epoch_max_accs[epoch_acc] = data_epoch
    top_keys = sorted(epoch_max_accs.keys(), reverse=True)[:5]
    epoch_max_accs = {k: epoch_max_accs[k] for k in top_keys}
    print(np.mean(np.array(accs)))
    writer.add_scalar("Loss/val", np.mean(np.array(losses)), epoch)
    writer.add_scalar("Accuracy/val", np.mean(np.array(accs)), epoch)

data_best_epochs = [item for sublist in epoch_max_accs.values() for item in sublist]
np.save('runs_nature/'+run_id+date_str+'/nature_eval_values.npy', data_best_epochs)