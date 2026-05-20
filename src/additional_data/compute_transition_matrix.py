import matplotlib.pyplot as plt
import numpy as np
import pickle

from tqdm import tqdm

# Load dataset
data_path = '../../DATA/DATA_APLO/'
file_all = data_path+'train_all.pkl'
file_target = data_path+'train_aplo.pkl'

with open(file_all, 'rb') as f_all, open(file_target, 'rb') as f_target:
    data_all = pickle.load(f_all)
    data_target = pickle.load(f_target)

seqs_a1 = []
seqs_target = []
for i, d in enumerate(data_all):
    for m_id, seq in data_all[i]:
        if 'A1_' in m_id:
            seqs_a1.append(seq)
    seqs_target.append(data_target[i][1])

unique_labels = [a for a in 'ARNDCEQGHILKMSPFTWYV']
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
transition_matrix = np.zeros((len(label_to_idx),len(label_to_idx)))

for i, _ in tqdm(enumerate(seqs_a1)):
    s_a1 = seqs_a1[i]
    s_target = seqs_target[i]
    for k, _ in enumerate(s_a1):
        if s_a1[k] != s_target[k]:
            transition_matrix[label_to_idx[s_a1[k]], label_to_idx[s_target[k]]] += 1

transition_matrix_norm = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

plt.matshow(transition_matrix_norm, cmap='viridis')  # or 'hot', 'plasma', 'coolwarm'
plt.colorbar()  # Show scale
plt.title('Transition matrix (normalized by line)')
plt.xlabel('A1')
plt.ylabel('Target species')
plt.yticks(ticks=range(len(unique_labels)), labels=unique_labels)
plt.xticks(ticks=range(len(unique_labels)), labels=unique_labels, rotation=45)
plt.savefig('transition_matrix.png')

transition_dict = {}
for i, label in enumerate(unique_labels):
    transition_dict[label] = transition_matrix_norm[i]

with open('transition_matrix_dict.pkl', 'wb') as f:
    pickle.dump(transition_dict, f)