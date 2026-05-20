import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Parsing
parser = argparse.ArgumentParser(description="Visualization of position")
parser.add_argument("-f", "--file", type=str, help="File to analyse")

args = parser.parse_args()
nature_values = np.load(args.file, allow_pickle=True)


unique_labels = [a for a in '-ARNDCEQGHILKMSPFTWYV']
unique_labels_nogap = [a for a in 'ARNDCEQGHILKMSPFTWYV']
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}


def accuracy(nature_values):
    accs = []
    tot, pos = 0, 0
    for d in nature_values:
        positions = d['positions']
        labels = d['labels']
        labels = labels[positions]
        preds = d['preds']
        amax = np.argmax(np.array(preds), axis=1)
        amax = amax[positions[0]]
        sim = [labels[i] == amax[i] for i in range(len(amax))]
        tot += len(sim)
        pos += sum(sim)
        if len(sim) > 0:
            accs.append(sum(sim)/len(sim))
    return sum(accs)/len(accs), pos/tot



def accuracy_by_subst_rate(nature_values):
    all_subst_rates = []
    tot, pos = 0, 0
    for d in nature_values:
        positions = d['positions'][0]
        if sum(positions) > 0:
            all_subst_rates.append(sum(positions)/len(positions))
            tot += len(positions)
            pos += sum(positions)
    median_subst_rate = np.median(np.array(all_subst_rates))
    print('median_subst_rate:', median_subst_rate)
    print('mean_subst_rate:', pos/tot)

    accs_top, accs_bot = [], []
    tot_top, pos_top = 0, 0
    tot_bot, pos_bot = 0, 0
    for d in nature_values:
        positions = d['positions']
        labels = d['labels']
        labels = labels[positions]
        preds = d['preds']
        amax = np.argmax(np.array(preds), axis=1)
        amax = amax[positions[0]]
        sim = [labels[i] == amax[i] for i in range(len(amax))]
        subst_rate = sum(positions[0])/len(positions[0])
        if subst_rate > median_subst_rate:
            tot_top += len(sim)
            pos_top += sum(sim)
            if len(sim) > 0:
                accs_top.append(sum(sim)/len(sim))
        else:
            tot_bot += len(sim)
            pos_bot += sum(sim)
            if len(sim) > 0:
                accs_bot.append(sum(sim)/len(sim))
    return sum(accs_top)/len(accs_top), pos_top/tot_top, sum(accs_bot)/len(accs_bot), pos_bot/tot_bot


def accuracy_by_subst_rate_continuous(nature_values, nb_bins=500):
    accs, subst_rates = [], []
    for d in nature_values:
        positions = d['positions']
        labels = d['labels']
        labels = labels[positions]
        preds = d['preds']
        amax = np.argmax(np.array(preds), axis=1)
        amax = amax[positions[0]]
        sim = [labels[i] == amax[i] for i in range(len(amax))]
        if len(sim)>0:
            subst_rates.append(sum(positions[0])/len(positions[0]))
            accs.append(sum(sim)/len(sim))
    subst_rates, accs = zip(*sorted(zip(subst_rates, accs)))
    subst_rates, accs = np.array(subst_rates), np.array(accs)
    bins = [int(i*nb_bins/len(subst_rates)) for i in range(len(subst_rates))]
    subst_rates_binned, accs_binned = [], []
    for i in range(nb_bins):
        bin_mask = np.array([bins[j]==i for j in range(len(bins))])
        tmp_subst_rates = subst_rates[bin_mask]
        tmp_accs = accs[bin_mask]
        subst_rates_binned.append(sum(tmp_subst_rates)/len(tmp_subst_rates))
        accs_binned.append(sum(tmp_accs)/len(tmp_accs))
    return subst_rates_binned, accs_binned


def accuracy_by_position(nature_values, nb_bins = 100):
    tot, pos = np.zeros(nb_bins), np.zeros(nb_bins)
    for d in nature_values:
        positions = d['positions'][0]
        labels = d['labels'][0]
        preds = d['preds']
        preds = np.argmax(np.array(preds), axis=1)
        bins = [int(i*nb_bins/len(positions)) for i in range(len(positions))]
        for i in range(len(positions)):
            if positions[i] == 1:
                tot[bins[i]] += 1
                if labels[i] == preds[i]:
                    pos[bins[i]] += 1
    return np.array(pos)/np.array(tot)


def confusion_matrix_nature(nature_values):
    all_labels, all_pred = [], []
    for d in nature_values:
        positions = d['positions']
        labels = d['labels']
        labels = labels[positions]
        preds = d['preds']
        amax = np.argmax(np.array(preds), axis=1)
        amax = amax[positions[0]]
        pred = amax
        all_labels.extend(labels)
        all_pred.extend(pred)
    cm = confusion_matrix(all_labels, all_pred, labels=range(len(unique_labels)))
    cm = cm[1:,1:]
    row_sums = cm.sum(axis=1, keepdims=True) + 1e-8
    cm_norm = cm / row_sums
    return cm_norm


def accuracy_matrix(nature_values):
    all_labels, all_pred, all_a1 = [], [], []
    for d in nature_values:
        positions = d['positions']
        labels = d['labels']
        labels = labels[positions]
        a1 = d['a1']
        a1 = a1[positions]
        preds = d['preds']
        amax = np.argmax(np.array(preds), axis=1)
        amax = amax[positions[0]]
        pred = amax
        all_labels.extend(labels)
        all_pred.extend(pred)
        all_a1.extend(a1)
    matrix_pos = np.zeros((len(unique_labels), len(unique_labels)))
    matrix_tot = np.zeros((len(unique_labels), len(unique_labels)))
    for i, l in tqdm(enumerate(all_labels)):
        matrix_tot[all_a1[i], all_labels[i]] += 1
        if all_labels[i]==all_pred[i] and all_a1[i]!=all_labels[i]:
            matrix_pos[all_a1[i], all_labels[i]] += 1
    matrix_accuracy = matrix_pos[1:,1:]/matrix_tot[1:,1:]    # remove gap values
    return matrix_accuracy


def most_common_substitutions_accuracy(nature_values):
    all_labels, all_pred, all_a1 = [], [], []
    for d in nature_values:
        positions = d['positions']
        labels = d['labels']
        labels = labels[positions]
        a1 = d['a1']
        a1 = a1[positions]
        preds = d['preds']
        amax = np.argmax(np.array(preds), axis=1)
        amax = amax[positions[0]]
        pred = amax
        all_labels.extend(labels)
        all_pred.extend(pred)
        all_a1.extend(a1)
    matrix_pos = np.zeros((len(unique_labels), len(unique_labels)))
    matrix_tot = np.zeros((len(unique_labels), len(unique_labels)))
    for i, l in tqdm(enumerate(all_labels)):
        matrix_tot[all_a1[i], all_labels[i]] += 1
        if all_labels[i]==all_pred[i] and all_a1[i]!=all_labels[i]:
            matrix_pos[all_a1[i], all_labels[i]] += 1
    list_pos = matrix_pos[1:,1:].flatten().tolist()
    list_tot = matrix_tot[1:,1:].flatten().tolist()
    sorted_pairs = sorted(zip(list_tot, list_pos), reverse=True)
    list_tot, list_pos = zip(*sorted_pairs)
    all_accuracies = []
    all_percent = []
    for i in range(1, len(list_tot)):
        acc = sum(list_pos[:i]) / sum(list_tot[:i])
        percent = sum(list_tot[:i]) / sum(list_tot)
        all_accuracies.append(acc)
        all_percent.append(percent)
    return all_accuracies, all_percent




print('> computing accuracies by most common substitutions')
accs, percent = most_common_substitutions_accuracy(nature_values)
plt.figure(figsize=(8, 6))
plt.plot(percent, accs)
plt.title("Accuracy by most common substitutions")
plt.xlabel("% of most common substitutuions")
plt.ylabel("Accuracy")
plt.show()
plt.savefig('accuracy_by_most_common_substitutions.svg')





print('> computing confusion matrix')
cm = confusion_matrix_nature(nature_values)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap="magma")
plt.title("Confusion Matrix")
plt.colorbar()
tick_labels = unique_labels[1:]  # or ["Class 0", "Class 1", "Class 2"]
tick_marks = range(len(tick_labels))
plt.xticks(tick_marks, tick_labels)
plt.yticks(tick_marks, tick_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
plt.savefig('confusion_matrix.svg')


print('> computing accuracy matrix')
mat = accuracy_matrix(nature_values)
plt.figure(figsize=(8, 6))
plt.imshow(mat, interpolation="nearest", cmap="magma")
plt.title("Confusion Matrix")
plt.colorbar()
tick_labels = unique_labels[1:]  # or ["Class 0", "Class 1", "Class 2"]
tick_marks = range(len(tick_labels))

print('> computing accuracy')
plt.xticks(tick_marks, tick_labels)
plt.yticks(tick_marks, tick_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
plt.savefig('accuracy_matrix.svg')


print('> computing accuracy')
mean_of_accuracies, mean_accuracy = accuracy(nature_values)
print('mean_of_accuracies:', mean_of_accuracies)
print('mean_accuracy:', mean_accuracy)

print('> computing accuracy')
mean_of_accuracies_top, mean_accuracy_top, mean_of_accuracies_bot, mean_accuracy_bot = accuracy_by_subst_rate(nature_values)
print('mean_of_accuracies for sequences with 50% high subst rate:', mean_of_accuracies_top)
print('mean_accuracy for sequences with 50% high subst rate:', mean_accuracy_top)
print('mean_of_accuracies for sequences with 50% low subst rate:', mean_of_accuracies_bot)
print('mean_accuracy for sequences with 50% low subst rate:', mean_accuracy_bot)


print('> computing accuracy by subst rate')
subst_rates, accs = accuracy_by_subst_rate_continuous(nature_values, nb_bins=250)
plt.figure(figsize=(8, 6))
plt.plot(subst_rates, accs)
plt.title("Accuracy by substitution rate")
plt.xlabel("Substitution rate")
plt.ylabel("Accuracy")
plt.xscale('log')
plt.show()
plt.savefig('accuracy_by_subst_rate.svg')


print('> computing accuracy by position')
nb_bins = 100
accs = accuracy_by_position(nature_values, nb_bins=nb_bins)
plt.figure(figsize=(8, 6))
plt.plot(np.array(list(range(nb_bins)))/nb_bins, accs)
plt.title("Accuracy by position")
plt.xlabel("Position (%)")
plt.ylabel("Accuracy")
plt.show()
plt.savefig('accuracy_by_position.svg')


