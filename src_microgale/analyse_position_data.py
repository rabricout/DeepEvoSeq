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
    scores = []
    for epoch_data in tqdm(position_values):
        y_true_epoch = []
        y_pred_epoch = []
        for d in epoch_data:
            y_true_epoch.append(d[key1])
            y_pred_epoch.append([int(p > 0.5) for p in d[key2]])
        f1_score_epoch = f1_score(np.concatenate(y_true_epoch), np.concatenate(y_pred_epoch))
        scores.append(f1_score_epoch)
    return scores

def youdens_J(position_values, key1="labels", key2="preds"):
    scores = []
    for epoch_data in tqdm(position_values):
        y_true_epoch = []
        y_pred_epoch = []
        for d in epoch_data:
            y_true_epoch.append(d[key1])
            y_pred_epoch.append([int(p > 0.5) for p in d[key2]])
        tn, fp, fn, tp = confusion_matrix(np.concatenate(y_true_epoch), np.concatenate(y_pred_epoch)).ravel()
        J = f_youden_j(tp, fp, tn, fn)
        scores.append(J)
    return scores

def phi(position_values, key1="labels", key2="preds"):
    scores = []
    for epoch_data in tqdm(position_values):
        y_true_epoch = []
        y_pred_epoch = []
        for d in epoch_data:
            y_true_epoch.append(d[key1])
            y_pred_epoch.append([int(p > 0.5) for p in d[key2]])
        mcc = matthews_corrcoef(np.concatenate(y_true_epoch), np.concatenate(y_pred_epoch))
        scores.append(mcc)
    return scores

def intermediate_values(position_values, thresh=0.5, key1="labels", key2="preds"):
    precision, recall, fpr, tpr = [], [], [], []
    for epoch_data in position_values:
        y_true_epoch = []
        y_pred_epoch = []
        for d in epoch_data:
            y_true_epoch.append(d[key1])
            y_pred_epoch.append([int(p > thresh) for p in d[key2]])
        y_true = np.concatenate(y_true_epoch)
        y_pred = np.concatenate(y_pred_epoch)
        precision.append(precision_score(y_true, y_pred))
        recall.append(recall_score(y_true, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
    return precision, recall, fpr, tpr



print('> computing f-score')
f_scores_values = f_scores(position_values)
print('f_scores_values', f_scores_values)
plt.plot(f_scores_values)
plt.xlabel("epochs")
plt.ylabel("f1-score")
plt.title("f1-score")
plt.savefig(output_path/"f1_score.png")
plt.clf()


print('> computing Youden J')
youdens_J = youdens_J(position_values)
print('youdens_J', youdens_J)
plt.plot(youdens_J)
plt.xlabel("epochs")
plt.ylabel("Youden's J")
plt.title("Youden's J")
plt.savefig(output_path/"youdens_J.png")
plt.clf()


print('> computing phi')
phi_values = phi(position_values)
print('phi_values', phi_values)
plt.plot(phi_values)
plt.xlabel("epochs")
plt.ylabel("phi")
plt.title("phi")
plt.savefig(output_path/"phi.png")
plt.clf()


if False:
    print('> computing intermediate values')
    precision, recall, fpr, tpr = intermediate_values(position_values)
    plt.scatter(recall, precision)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("precision/recall")
    plt.savefig("precision_recall.png")
    plt.clf()

    plt.scatter(fpr, tpr)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.savefig("ROC.png")
    plt.clf()


print('> doing continuous precision/recall and ROC')
precision, recall, fpr, tpr = [], [], [], []
for t in tqdm(np.arange(0, 1.01, 0.1)):
    precision_t, recall_t, fpr_t, tpr_t = intermediate_values(position_values, thresh=t)
    recall.append(np.mean(np.array(recall_t)[-10:]))
    precision.append(np.mean(np.array(precision_t)[-10:]))
    fpr.append(np.mean(np.array(fpr_t)[-10:]))
    tpr.append(np.mean(np.array(tpr_t)[-10:]))
mid = int(len(recall)/2)
print("recall, precision", recall[mid], precision[mid])
plt.plot(recall, precision)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("precision/recall")
plt.savefig(output_path/"precision_recall.png")
plt.clf()

plt.plot(fpr, tpr)
mid = int(len(fpr)/2)
print("fpr, tpr", fpr[mid], tpr[mid])
plt.plot(fpr, tpr)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.savefig(output_path/"ROC.png")
plt.clf()