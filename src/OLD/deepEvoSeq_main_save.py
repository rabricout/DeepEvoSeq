import numpy as np
import torch
import esm

import torch.nn.functional as F

from torch.utils.data import DataLoader
from deepEvoSeq_models import DeepEvoSeqAttnHead
from deepEvoSeq_models import DeepEvoSeqAttnHeadPosition
from deepEvoSeq_dataset import DeepEvoSeqDataset
from deepEvoSeq_dataset import DeepEvoSeqDatasetPosition
from utils import collate_fn_aa
from tqdm import tqdm

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

BATCH_SIZE=16
NUM_WORKERS=4

position = True
if position:
    train_dataset = DeepEvoSeqDatasetPosition(label_to_idx, data_path+'train_a1.pkl', data_path+'train_aplo.pkl')
    eval_dataset = DeepEvoSeqDatasetPosition(label_to_idx, data_path+'eval_a1.pkl', data_path+'eval_aplo.pkl')
    model_deepEvoSeq = DeepEvoSeqAttnHeadPosition(320)
    model_deepEvoSeq = model_deepEvoSeq.to(device)
else:
    train_dataset = DeepEvoSeqDataset(label_to_idx, data_path+'train_a1.pkl', data_path+'train_aplo.pkl')
    eval_dataset = DeepEvoSeqDataset(label_to_idx, data_path+'eval_a1.pkl', data_path+'eval_aplo.pkl')
    nb_amino_acids = 21
    model_deepEvoSeq = DeepEvoSeqAttnHead(320, nb_amino_acids)
    model_deepEvoSeq = model_deepEvoSeq.to(device)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn = collate_fn_aa,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,  # if using GPU
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    collate_fn = collate_fn_aa,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,  # if using GPU
)


# Training loop
optimizer = torch.optim.Adam(model_deepEvoSeq.parameters(), lr=3e-4)
num_epochs = 10

for epoch in range(num_epochs):
    # Training
    print('> epoch', epoch)
    accs = []
    model_deepEvoSeq.train()
    for batch_x, batch_y in tqdm(train_loader):
        #batch_x = batch_x.to("cuda")
        batch_y = batch_y.to("cuda")
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_x)
        with torch.no_grad():
            batch_tokens = batch_tokens.to(device)
            esm_embedding = model_esm_body(batch_tokens, repr_layers=[6], return_contacts=False)
        reps = esm_embedding["representations"][6][:,1:-1:]  # layer 6 for this model
        reps = reps.to(device)
        logits = model_deepEvoSeq(reps)
        if position:
            loss = F.cross_entropy(
                logits.reshape(-1, 2),
                batch_y.reshape(-1),
                ignore_index=-100,
            )        
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, nb_amino_acids),
                batch_y.reshape(-1),
                ignore_index=-100,
            )
        acc = (logits.argmax(dim=-1) == batch_y).float().mean()
        accs.append(acc.cpu())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Mean training accuracy for epoch', epoch, np.mean(np.array(accs)))

    # Validation
    model_deepEvoSeq.eval()
    accs = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(eval_loader):
            # batch_x = batch_x.to("cuda")
            batch_y = batch_y.to("cuda")

            batch_labels, batch_strs, batch_tokens = batch_converter(batch_x)
            with torch.no_grad():
                batch_tokens = batch_tokens.to(device)
                esm_embedding = model_esm_body(batch_tokens, repr_layers=[6], return_contacts=False)
            reps = esm_embedding["representations"][6][:,1:-1:]  # layer 6 for this model
            reps = reps.to(device)
            logits = model_deepEvoSeq(reps)
            if position:
                loss = F.cross_entropy(
                    logits.reshape(-1, 2),
                    batch_y.reshape(-1),
                    ignore_index=-100,
                )        
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, nb_amino_acids),
                    batch_y.reshape(-1),
                    ignore_index=-100,
                )

            acc = (logits.argmax(dim=-1) == batch_y).float().mean()
            accs.append(acc.cpu())
    print('Mean eval accuracy for epoch', epoch, np.mean(np.array(accs)))
            # compute accuracy, store metrics, etc.

