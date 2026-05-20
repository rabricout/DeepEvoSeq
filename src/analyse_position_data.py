import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from pathlib import Path
from tqdm import tqdm

# Parsing
parser = argparse.ArgumentParser(description="Visualization of position")
parser.add_argument("-f", "--file", type=str, help="File to analyse")

args = parser.parse_args()
position_values = np.load(args.file, allow_pickle=True)
output_path = Path(args.file).parent/'figures'
Path(output_path).mkdir(parents=True, exist_ok=True)


def f_youden_j(tp, fp, tn, fn):
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sen + spec - 1

def f_scores(position_values, key1="labels", key2="preds"):
    y_true = []
    y_pred = []
    for d in position_values:
        y_true.append(d[key1])
        y_pred.append([int(p > 0.5) for p in d[key2]])
    return f1_score(np.concatenate(y_true), np.concatenate(y_pred))

def youdens_J(position_values, key1="labels", key2="preds"):
    scores = []
    y_true = []
    y_pred = []
    for d in tqdm(position_values):
        y_true.append(d[key1])
        y_pred.append([int(p > 0.5) for p in d[key2]])
    tn, fp, fn, tp = confusion_matrix(np.concatenate(y_true), np.concatenate(y_pred)).ravel()
    J = f_youden_j(tp, fp, tn, fn)
    return J

def phi(position_values, key1="labels", key2="preds"):
    scores = []
    y_true = []
    y_pred = []
    for d in tqdm(position_values):
        y_true.append(d[key1])
        y_pred.append([int(p > 0.5) for p in d[key2]])
    mcc = matthews_corrcoef(np.concatenate(y_true), np.concatenate(y_pred))
    return mcc

def intermediate_values(position_values, thresh=0.5, key1="labels", key2="preds"):
    precision, recall, fpr, tpr = [], [], [], []
    y_true = []
    y_pred = []
    for d in position_values:
        y_true.append(d[key1])
        y_pred.append([int(p > thresh) for p in d[key2]])
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    return precision, recall, fpr, tpr



print('> computing f-score')
f_score_value = f_scores(position_values)
print('f_scores_values', f_score_value)


print('> computing Youden J')
youdens_J = youdens_J(position_values)
print('youdens_J', youdens_J)


print('> computing phi')
phi_value = phi(position_values)
print('phi_value', phi_value)


position_values_random = np.load('runs_position/Random/position_eval_values.npy', allow_pickle=True)
position_values_proxy = np.load('runs_position/Proxy/position_eval_values.npy', allow_pickle=True)
position_values_species = np.load('runs_position/Species/position_eval_values.npy', allow_pickle=True)
position_values_consensus = np.load('runs_position/Consensus/position_eval_values.npy', allow_pickle=True)
baseline_values = [position_values_random, position_values_proxy, position_values_species, position_values_consensus]
baseline_names = ['Random', 'Proxy', 'Species', 'Consensus']

print('> doing continuous precision/recall and ROC')
precision, recall, fpr, tpr = [], [], [], []
for t in tqdm(np.arange(0, 1.01, 0.05)):
    precision_t, recall_t, fpr_t, tpr_t = intermediate_values(position_values, thresh=t)
    recall.append(recall_t)
    precision.append(precision_t)
    fpr.append(fpr_t)
    tpr.append(tpr_t)
# mid = int(len(recall)/2)
for i, name in enumerate(baseline_names):
    precision_t, recall_t, fpr_t, tpr_t = intermediate_values(baseline_values[i], thresh=0.5)
    plt.scatter(recall_t, precision_t, label=baseline_names[i])
# print("recall, precision", recall[mid], precision[mid])
plt.plot(recall, precision, label='DeepEvoSeq')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("recall")
plt.ylabel("precision")
plt.legend()
plt.title("precision/recall")
plt.savefig(output_path/"precision_recall.svg")
plt.clf()

plt.plot(fpr, tpr)
mid = int(len(fpr)/2)
print("fpr, tpr", fpr[mid], tpr[mid])
for i, values in enumerate(baseline_names):
    precision_t, recall_t, fpr_t, tpr_t = intermediate_values(baseline_values[i], thresh=0.5)
    plt.scatter(fpr_t, tpr_t, label=baseline_names[i])
plt.plot(fpr, tpr, label='DeepEvoSeq')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()
plt.plot([0, 1], [0, 1], linestyle='--', color='black')
plt.title("ROC curve")
plt.savefig(output_path/"ROC.svg")
plt.clf()