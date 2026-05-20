import random
import pickle
import torch

from torch.utils.data import Dataset
from tqdm import tqdm


class DeepEvoSeqDataset(Dataset):
    def __init__(self, label_to_idx, data_path: str, labels_path: str):
        self.label_to_idx = label_to_idx
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)

        assert len(self.data) == len(self.labels), "Data and labels must have same length"

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
    


class DeepEvoSeqDatasetPosition(Dataset):
    def __init__(self, label_to_idx, data_path: str, labels_path: str):
        self.label_to_idx = label_to_idx
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            labels = pickle.load(f)

        assert len(self.data) == len(labels), "Data and labels must have same length"
        data_a1 = []
        data_no_a1 = []
        for i, d in enumerate(self.data):
            datum_no_a1 = []
            for m_id, s in d:
                if 'A1_' in m_id:
                    data_a1.append(s)
                else:
                    datum_no_a1.append((m_id, s))
            data_no_a1.append(datum_no_a1)
        positions = []
        # Building substitution mask for the position task
        for i, yi in enumerate(labels):
            seq_target = labels[i][1]
            seq_a1 = data_a1[i]
            p = [1 if seq_target[k]!=seq_a1[k] else 0 for k in range(len(seq_target))]
            positions.append(p)
        self.labels = positions
        self.data = data_no_a1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # e.g., a tensor or numpy array of shape (L, D)
        y = self.labels[idx] # e.g., int or tensor of shape (L,) or (L,)
        #y = [self.label_to_idx[label] for label in y[1]]
        # x_list = []
        # for x_i in x:
        #     x_list.append([self.label_to_idx[data] for data in x_i[1]])
        # x = torch.tensor(x_list)
        # Convert to torch tensors if needed
        # if not torch.is_tensor(x):
        #     x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)

        return x, y
    

class DeepEvoSeqSimpleDatasetPosition(Dataset):
    def __init__(self, label_to_idx, data_path: str, labels_path: str):
        self.label_to_idx = label_to_idx
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            labels = pickle.load(f)

        assert len(self.data) == len(labels), "Data and labels must have same length"
        data_a1 = []
        for i, d in enumerate(self.data):
            for m_id, s in d:
                if 'A1_' in m_id:
                    data_a1.append(s)
        positions = []
        # Building substitution mask for the position task
        for i, yi in enumerate(labels):
            seq_target = labels[i][1]
            seq_a1 = data_a1[i]
            p = [1 if seq_target[k]!=seq_a1[k] else 0 for k in range(len(seq_target))]
            positions.append(p)
        self.labels = positions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # e.g., a tensor or numpy array of shape (L, D)
        y = self.labels[idx] # e.g., int or tensor of shape (L,) or (L,)
        #y = [self.label_to_idx[label] for label in y[1]]
        x_list = []
        for x_i in x:
            x_list.append([self.label_to_idx[data] for data in x_i[1]])
        x = torch.tensor(x_list)
        # Convert to torch tensors if needed
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)

        return x, y
    


class DeepEvoSeqDatasetNature(Dataset):
    def __init__(self, label_to_idx, data_path: str, labels_path: str, window: int=-1):
        self.label_to_idx = label_to_idx
        self.window = window
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)

        assert len(self.data) == len(self.labels), "Data and labels must have same length"
        data_a1 = []
        data_no_a1 = []
        for i, d in enumerate(self.data):
            datum_no_a1 = []
            for m_id, s in d:
                if 'A1_' in m_id:
                    data_a1.append(s)
                else:
                    datum_no_a1.append((m_id, s))
            data_no_a1.append(datum_no_a1)
        positions = []
        # Building substitution mask for the position task
        for i, yi in enumerate(self.labels):
            seq_target = self.labels[i][1]
            seq_a1 = data_a1[i]
            p = [seq_target[k]!=seq_a1[k] for k in range(len(seq_target))]
            positions.append(p)
        self.positions = positions
        self.data = data_no_a1
        self.data_a1 = data_a1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # e.g., a tensor or numpy array of shape (L, D)
        x_a1 = self.data_a1[idx]   # e.g., a tensor or numpy array of shape (L, D)
        y = self.labels[idx] # e.g., int or tensor of shape (L,) or (L,)
        p = self.positions[idx]
        y = [self.label_to_idx[label] for label in y[1]]
        x_a1 = [self.label_to_idx[d] for d in x_a1]
        x_list = []
        for x_i in x:
            x_list.append([self.label_to_idx[data] for data in x_i[1]])
        x_tensor = torch.tensor(x_list)

        # Convert to torch tensors if needed
        # if not torch.is_tensor(x):
        #     x = torch.tensor(x)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        if not torch.is_tensor(x_a1):
            x_a1 = torch.tensor(x_a1, dtype=torch.long)
        if not torch.is_tensor(p):
            p = torch.tensor(p, dtype=torch.bool)
        
        if self.window != -1:
            seq_len = x.shape[1]
            random_start = random.randint(0, seq_len-self.window)
            x_tensor = x_tensor[:,random_start:random_start+self.window]
            x = x[:,random_start:random_start+self.window]
            x_a1 = x_a1[random_start:random_start+self.window]
            y = y[random_start:random_start+self.window]
            p = p[random_start:random_start+self.window]

        return (x_tensor, x, x_a1), (y, p)



class DeepEvoSeqSimpleDatasetNature(Dataset):
    def __init__(self, label_to_idx, data_path: str, labels_path: str, window: int=-1):
        self.label_to_idx = label_to_idx
        self.window = window
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)

        assert len(self.data) == len(self.labels), "Data and labels must have same length"
        data_a1 = []
        data_no_a1 = []
        for i, d in enumerate(self.data):
            datum_no_a1 = []
            for m_id, s in d:
                if 'A1_' in m_id:
                    data_a1.append(s)
                else:
                    datum_no_a1.append((m_id, s))
            data_no_a1.append(datum_no_a1)
        positions = []
        # Building substitution mask for the position task
        for i, yi in enumerate(self.labels):
            seq_target = self.labels[i][1]
            seq_a1 = data_a1[i]
            #seq_a1 = self.data[i][1]
            p = [seq_target[k]!=seq_a1[k] for k in range(len(seq_target))]
            positions.append(p)
        self.positions = positions
        self.data = data_no_a1
        self.data_a1 = data_a1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # e.g., a tensor or numpy array of shape (L, D)
        x_a1 = self.data_a1[idx]   # e.g., a tensor or numpy array of shape (L, D)
        y = self.labels[idx] # e.g., int or tensor of shape (L,) or (L,)
        p = self.positions[idx]
        x_list = []
        for x_i in x:
            x_list.append([self.label_to_idx[data] for data in x_i[1]])
        x = torch.tensor(x_list)
        x_a1 = [self.label_to_idx[d] for d in x_a1]
        y = [self.label_to_idx[label] for label in y[1]]

        # Convert to torch tensors if needed
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not torch.is_tensor(x_a1):
            x_a1 = torch.tensor(x_a1)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.long)
        if not torch.is_tensor(p):
            p = torch.tensor(p, dtype=torch.bool)
        
        if self.window != -1:
            seq_len = x.shape[1]
            random_start = random.randint(0, seq_len-self.window)
            x = x[:,random_start:random_start+self.window]
            x_a1 = x_a1[random_start:random_start+self.window]
            y = y[random_start:random_start+self.window]
            p = p[random_start:random_start+self.window]

        return (x, x_a1), (y, p)
    


class DeepEvoSeqDatasetNatureSubstRate(Dataset):
    def __init__(self, label_to_idx, data_path: str, labels_path: str):
        self.label_to_idx = label_to_idx
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)

        assert len(self.data) == len(self.labels), "Data and labels must have same length"
        data_a1 = []
        for i, d in enumerate(self.data):
            for m_id, s in d:
                if 'A1_' in m_id:
                    data_a1.append(s)
        subst_rate = []
        for i, yi in enumerate(self.labels):
            seq_target = self.labels[i][1]
            seq_a1 = data_a1[i]
            r = [seq_target[k]!=seq_a1[k] for k in range(len(seq_target))]
            subst_rate.append(sum(r)/len(r))
        self.labels = subst_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # e.g., a tensor or numpy array of shape (L, D)
        y = self.labels[idx] # e.g., int or tensor of shape (L,) or (L,)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        return x, y



class DeepEvoSeqSimpleDatasetNatureSubstRate(Dataset):
    def __init__(self, label_to_idx, data_path: str, labels_path: str):
        self.label_to_idx = label_to_idx
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        with open(labels_path, "rb") as f:
            self.labels = pickle.load(f)

        assert len(self.data) == len(self.labels), "Data and labels must have same length"
        data_a1 = []
        for i, d in enumerate(self.data):
            for m_id, s in d:
                if 'A1_' in m_id:
                    data_a1.append(s)
        all_subst_rates = []
        for i, yi in tqdm(enumerate(self.labels)):
            tmp_subst_rates = []
            datum = self.data[i]
            for j1, yi in enumerate(datum):
                for j2, yj in enumerate(datum):
                    r = [self.data[i][j1][1][k]!=self.data[i][j2][1][k] for k in range(len(self.data[i][j1][1]))]
                    tmp_subst_rates.append(sum(r)/len(r))
            all_subst_rates.append(tmp_subst_rates)
        self.all_subst_rates = all_subst_rates
        subst_rate = []
        for i, yi in enumerate(self.labels):
            seq_target = self.labels[i][1]
            seq_a1 = data_a1[i]
            r = [seq_target[k]!=seq_a1[k] for k in range(len(seq_target))]
            subst_rate.append(sum(r)/len(r))
        self.labels = subst_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]   # e.g., a tensor or numpy array of shape (L, D)
        x2 = self.all_subst_rates[idx]
        y = self.labels[idx] # e.g., int or tensor of shape (L,) or (L,)
        x_list = []
        for x_i in x:
            x_list.append([self.label_to_idx[data] for data in x_i[1]])
        x = torch.tensor(x_list)

        # Convert to torch tensors if needed
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        # if not torch.is_tensor(x2):
        #     x2 = torch.tensor(x2)
        if not torch.is_tensor(y):
            y = torch.tensor(y)
        return (x, x2), y
    

