### This script compares the different reconstruction methods. 
### It computes similarity matrices
### It also computes the number of sequences that are totally similar

import matplotlib.pyplot as plt
import numpy as np
import os

from Bio import SeqIO
from tqdm import tqdm


path = 'DATA_ALIGNED/'

method_id = {'JC69':0, 'LG08':1, 'LG08_F':2, 'WAG01_F':3}
matrix = np.zeros((4,4))
matrix_identical = np.zeros((4,4))


def compare_sequences(s1,s2):
    diff = 0
    for i, a in enumerate(s1):
        if s1[i] != s2[i]:
            diff += 1
    return diff/len(s2)

sequences = {'JC69': [], 'LG08': [], 'LG08_F': [], 'WAG01_F': []}
for filename in tqdm(os.listdir(path)):
    records = list(SeqIO.parse(path+filename, "fasta"))
    if len(records) == 4:
        for record in records:
            if 'JC69' in record.id:
                sequences['JC69'].append(str(record.seq))
            if 'LG08_' in record.id:
                sequences['LG08'].append(str(record.seq))
            if "LG08+F" in str(record.id):
                sequences['LG08_F'].append(str(record.seq))
            if 'WAG01+F' in record.id:
                sequences['WAG01_F'].append(str(record.seq))

for m1 in sequences.keys():
    for m2 in tqdm(sequences.keys()):
        values = []
        if m1 != m2:
            for i, s1 in tqdm(enumerate(sequences[m1])):
                v = compare_sequences(sequences[m1][i], sequences[m2][i])
                values.append(v)
                if v == 0:
                    matrix_identical[method_id[m1], method_id[m2]] += 1
            matrix[method_id[m1], method_id[m2]] = np.mean(np.array(values))

matrix_identical /= len(sequences['JC69'])
np.fill_diagonal(matrix_identical, 1)


from pathlib import Path
path_JC69 = Path('../BPPANCESTOR_JC69/')
path_LG08 = Path('../BPPANCESTOR_LG08/')
path_LG08_F = Path('../BPPANCESTOR_LG08_F/')
path_WAG01_F = Path('../BPPANCESTOR_WAG01_F/')

paths = {'JC69': path_JC69, 'LG08': path_LG08, 'LG08_F': path_LG08_F, 'WAG01_F': path_WAG01_F}
mean_values = {}

for k, v in paths.items():
    files = [f.name for f in Path(v/'DATA_FINAL_A1').iterdir() if f.is_file()]
    values = []
    for f in tqdm(files):
        records_A1 = []
        records_aplo = []
        for i, record in enumerate(SeqIO.parse(v/Path('DATA_FINAL_A1')/f, "fasta")):
            records_A1.append(record.seq)
        for i, record in enumerate(SeqIO.parse(v/Path('DATA_FINAL_APLO')/f, "fasta")):
            records_aplo.append(record.seq)
        if len(records_A1) > 1 or len(records_aplo) > 1:
            print('MORE THAN ONE RECORD, ERROR')
        values.append(compare_sequences(records_A1[0], records_aplo[0]))
    mean_value = np.mean(np.array(values))
    mean_values[k] = mean_value
average_a1_aplo_rate = np.mean(np.array(list(mean_values.values())))



# Display
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(matrix, cmap='hot', interpolation='nearest')
# Add numbers to each cell
for (j, i), val in np.ndenumerate(matrix):
    ax.text(i, j, f'{val:.5f}', ha='center', va='center', 
            fontsize=16, fontweight='bold',
            color='white' if abs(val) < np.max(matrix)/2 else 'black')
ax.set_xticks(np.arange(len(method_id.keys())))
ax.set_yticks(np.arange(len(method_id.keys())))
ax.set_xticklabels(method_id.keys(), fontsize=16)
ax.set_yticklabels(method_id.keys(), fontsize=16)
fig.colorbar(im, ax=ax)
plt.title("Average difference between reconstruction methods", fontsize=16)
plt.tight_layout()
plt.savefig('FIGURES/difference_between_reconstruction_methods.svg')
plt.clf()

# Display
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(matrix/average_a1_aplo_rate, cmap='hot', interpolation='nearest')
# Add numbers to each cell
for (j, i), val in np.ndenumerate(matrix/average_a1_aplo_rate):
    ax.text(i, j, f'{val*100:.2f}%', ha='center', va='center', 
            fontsize=16, fontweight='bold',
            color='white' if abs(val) < np.max(matrix/average_a1_aplo_rate)/2 else 'black')
ax.set_xticks(np.arange(len(method_id.keys())))
ax.set_yticks(np.arange(len(method_id.keys())))
ax.set_xticklabels(method_id.keys(), fontsize=16)
ax.set_yticklabels(method_id.keys(), fontsize=16)
fig.colorbar(im, ax=ax)
plt.title("Average difference divided by subst rate between A1 and aplodontia (%)", fontsize=16)
plt.tight_layout()
plt.savefig('FIGURES/difference_between_reconstruction_methods_normalized.svg')
plt.clf()

# Display
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(matrix_identical, cmap='hot', interpolation='nearest')
# Add numbers to each cell
for (j, i), val in np.ndenumerate(matrix_identical):
    ax.text(i, j, f'{val:.3f}', ha='center', va='center', 
            fontsize=16, fontweight='bold',
            color='white' if abs(val) < np.max(matrix_identical)/2 else 'black')
ax.set_xticks(np.arange(len(method_id.keys())))
ax.set_yticks(np.arange(len(method_id.keys())))
ax.set_xticklabels(method_id.keys(), fontsize=16)
ax.set_yticklabels(method_id.keys(), fontsize=16)
fig.colorbar(im, ax=ax)
plt.title("Fraction of sequences with no difference between the reconstruction methods", fontsize=16)
plt.tight_layout()
plt.savefig('FIGURES/identical_reconstructions.svg')
plt.clf()