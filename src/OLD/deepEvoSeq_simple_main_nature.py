import numpy as np
import torch
import esm

import torch.nn.functional as F

from torch.utils.data import DataLoader
from deepEvoSeq_simple_models import *
from deepEvoSeq_simple_dataset import DeepEvoSeqSimpleDataset
from utils import collate_fn_aa_position_simple
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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

train_dataset = DeepEvoSeqSimpleDataset(label_to_idx, data_path+'train_a1.pkl', data_path+'train_aplo.pkl')
eval_dataset = DeepEvoSeqSimpleDataset(label_to_idx, data_path+'eval_a1.pkl', data_path+'eval_aplo.pkl')
nb_amino_acids = len(label_to_idx)
#model_deepEvoSeq = DeepEvoSeqAttnHeadNature(320, nb_amino_acids)
model_deepEvoSeq = DeepEvoSeqSimpleNature(nb_amino_acids, nb_amino_acids)
model_deepEvoSeq = model_deepEvoSeq.to(device)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn = collate_fn_aa_position_simple,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,  # if using GPU
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    collate_fn = collate_fn_aa_position_simple,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,  # if using GPU
)


# Training loop
optimizer = torch.optim.Adam(model_deepEvoSeq.parameters(), lr=2e-5)
num_epochs = 500
date_str = datetime.now().strftime("%m_%d_%Y_%H_%M")
writer = SummaryWriter("runs_nature/"+'simple_'+date_str)

for epoch in range(num_epochs):
    # Training
    print('> epoch', epoch)
    model_deepEvoSeq.train()
    for batch_x, (batch_y, batch_p) in tqdm(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model_deepEvoSeq(batch_x)

        batch_y_masked = batch_y.clone()
        batch_y_masked[~batch_p] = -100  # -100 is ignore_index
        loss = F.cross_entropy(
            logits.reshape(-1, nb_amino_acids),
            batch_y_masked.reshape(-1),
            ignore_index=-100,
        )
        acc = (logits[batch_p].argmax(dim=-1) == batch_y[batch_p]).float().mean()
        writer.add_scalar("Loss/train", loss.cpu(), epoch)
        writer.add_scalar("Accuracy/train", acc.cpu(), epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model_deepEvoSeq.eval()
    accs = []
    losses = []
    with torch.no_grad():
        for batch_x, (batch_y, batch_p) in tqdm(eval_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model_deepEvoSeq(batch_x)

            batch_y_masked = batch_y.clone()
            batch_y_masked[~batch_p] = -100  # -100 is ignore_index
            loss = F.cross_entropy(
                logits.reshape(-1, nb_amino_acids),
                batch_y_masked.reshape(-1),
                ignore_index=-100,
            )
            losses.append(loss.cpu())
            acc = (logits[batch_p].argmax(dim=-1) == batch_y[batch_p]).float().mean()
            accs.append(acc.cpu())
    writer.add_scalar("Loss/val", np.mean(np.array(losses)), epoch)
    writer.add_scalar("Accuracy/val", np.mean(np.array(accs)), epoch)
