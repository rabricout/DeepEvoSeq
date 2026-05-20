import numpy as np
import argparse
import torch
import json
import esm

import torch.nn.functional as F

from torch.utils.data import DataLoader
from deepEvoSeq_models_position import *
from deepEvoSeq_dataset import *
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# Parsing
parser = argparse.ArgumentParser(description="DeepEvoSeq options")
parser.add_argument("-r", "--lr", type=float, help="Learning rate", default=2e-4)
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


# Prepare sequences: list of (sequence_id, description, sequence)
data_path = 'DATA/DATA_APLO/'
print('> constructing datasets and dataloaders')
BATCH_SIZE=32
NUM_WORKERS=4
# if is_simple:
#     train_dataset = DeepEvoSeqDatasetPosition(label_to_idx, data_path+'train_all.pkl', data_path+'train_aplo.pkl')
#     eval_dataset = DeepEvoSeqDatasetPosition(label_to_idx, data_path+'eval_all.pkl', data_path+'eval_aplo.pkl')
# else:
train_dataset = DeepEvoSeqDatasetPosition(label_to_idx, data_path+'train_all.pkl', data_path+'train_aplo.pkl')
eval_dataset = DeepEvoSeqDatasetPosition(label_to_idx, data_path+'eval_all.pkl', data_path+'eval_aplo.pkl')
nb_amino_acids = 21
nb_species = 4


model_deepEvoSeq = DeepEvoSeqPositionGeneric(nb_species, nb_amino_acids, attention=use_attention, is_simple=is_simple, attn_heads=8)
for name, module in model_deepEvoSeq.named_modules():
    print(name, "->", module)
input('...')
# if is_simple:
#     if use_attention:
#         model_deepEvoSeq = DeepEvoSeqSimpleAttnHeadPosition(nb_species=nb_species, nb_amino_acids=nb_amino_acids)
#     else:
#         model_deepEvoSeq = DeepEvoSeqSimpleFCPosition(nb_species=nb_species, nb_amino_acids=nb_amino_acids)
# else:
#     if use_attention:
#         model_deepEvoSeq = DeepEvoSeqAttnHeadPosition(nb_species=nb_species, esm_embed_dim=320)
#     else:
#         model_deepEvoSeq = DeepEvoSeqFCPosition(nb_species=nb_species, esm_embed_dim=320)
model_deepEvoSeq = model_deepEvoSeq.to(device)
collate_fn = collate_fn_aa_position


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
num_epochs = 200 if is_simple else 100
date_str = datetime.now().strftime("%m_%d_%Y_%H_%M")
run_id = ('simple_' if is_simple else 'full_') + str(lr) + '_'
writer = SummaryWriter("runs_position/"+run_id+date_str)
with open("runs_position/"+run_id+date_str+'/specs.txt', 'w', encoding='utf-8') as f:    # save model specs and training hyperparameters
    model_specs = model_deepEvoSeq.give_specs_dict()
    model_specs['attention'] = use_attention
    model_specs['simple'] = is_simple
    training_specs = {'lr':lr, 'batch_size':BATCH_SIZE, 'num_workers': NUM_WORKERS}
    json.dump([model_specs, training_specs], f, indent=2)

epoch_best_losses = {}
for epoch in range(num_epochs):
    # Training
    print('> epoch', epoch, '/', num_epochs)
    accs = []
    losses = []
    model_deepEvoSeq.train()
    for batch_x, batch_y in tqdm(train_loader):
        #batch_x = batch_x.to("cuda")
        batch_y = batch_y.to(device)
        if is_simple:
            batch_x_tensor, batch_x_raw = batch_x
            batch_x_tensor = batch_x_tensor.to(device)
            logits = model_deepEvoSeq(batch_x_tensor)
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

        loss = F.cross_entropy(
            logits.reshape(-1, 2),
            batch_y.reshape(-1),
            ignore_index=-100,
            weight=torch.tensor([1,10], device=device, dtype=torch.float),
        )
        acc = (logits.argmax(dim=-1) == batch_y).float().mean()
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
        for batch_x, batch_y in tqdm(eval_loader):
            # batch_x = batch_x.to("cuda")
            batch_y = batch_y.to(device)
            if is_simple:
                batch_x_tensor, batch_x_raw = batch_x
                batch_x_tensor = batch_x_tensor.to(device)
                logits = model_deepEvoSeq(batch_x_tensor)
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

            loss = F.cross_entropy(
                logits.reshape(-1, 2),
                batch_y.reshape(-1),
                ignore_index=-100,
                weight=torch.tensor([1,10], device=device, dtype=torch.float),
            )
            losses.append(loss.cpu())
            acc = (logits.argmax(dim=-1) == batch_y).float().mean()
            accs.append(acc.cpu())
            probs = F.softmax(logits[0], dim=1)
            data_epoch.append({'labels': np.array(batch_y[0].cpu()), 'preds': np.array(probs[:,1].cpu())})

    epoch_loss = np.mean(np.array(losses))
    epoch_best_losses[epoch_loss] = data_epoch
    top_5_keys = sorted(epoch_best_losses.keys())[:5]
    epoch_best_losses = {k: epoch_best_losses[k] for k in top_5_keys}
    print(np.mean(np.array(accs)))
    writer.add_scalar("Loss/val", np.mean(np.array(losses)), epoch)

data_best_5_epochs = [item for sublist in epoch_best_losses.values() for item in sublist]
np.save('runs_position/'+run_id+date_str+'/position_eval_values.npy', data_best_5_epochs)