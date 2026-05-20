import torch
import pickle
from torch.utils.data import Dataset

import torch.nn.functional as F


class DeepEvoSeqDatasetPosition(Dataset):
    def __init__(self, label_to_idx, data_path: str, labels_path: str):
        self.label_to_idx = label_to_idx
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            labels = pickle.load(f)

        assert len(self.data) == len(labels), "Data and labels must have same length"
        positions = []
        # Building substitution mask for the position task
        for i, yi in enumerate(labels):
            seq_target = labels[i][1]
            seq_a1 = self.data[i][1]
            p = [1 if seq_target[k]!=seq_a1[k] else 0 for k in range(len(seq_target))]
            positions.append(p)
        self.labels = positions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # e.g., a tensor or numpy array of shape (L, D)
        y = self.labels[idx] # e.g., int or tensor of shape (L,) or (L,)
        y = [self.label_to_idx[label] for label in y[1]]

        # Convert to torch tensors if needed
        # if not torch.is_tensor(x):
        #     x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)

        return x, y
    


class DeepEvoSeqSimpleDataset(Dataset):
    def __init__(self, label_to_idx, data_path: str, labels_path: str):
        self.label_to_idx = label_to_idx
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)

        assert len(self.data) == len(self.labels), "Data and labels must have same length"
        positions = []
        # Building substitution mask for the position task
        for i, yi in enumerate(self.labels):
            seq_target = self.labels[i][1]
            seq_a1 = self.data[i][1]
            p = [seq_target[k]!=seq_a1[k] for k in range(len(seq_target))]
            positions.append(p)
        self.positions = positions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # e.g., a tensor or numpy array of shape (L, D)
        y = self.labels[idx] # e.g., int or tensor of shape (L,) or (L,)
        p = self.positions[idx]
        x = torch.tensor([self.label_to_idx[data] for data in x[1]])
        x = F.one_hot(x, num_classes=len(self.label_to_idx)).float() 
        y = [self.label_to_idx[label] for label in y[1]]

        # Convert to torch tensors if needed
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        if not torch.is_tensor(p):
            p = torch.tensor(p, dtype=torch.bool)
        

        return x, (y, p)
    

