import io
import json
import scipy
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib

import residue_constants as residue_constants

from PIL import Image
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

matplotlib.use('TkAgg')


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()


def show_matrix_aa_to_image(mat, annot=False, normalize=None):
    mat = mat[:20,:20]
    if normalize=='pred':
        mat = mat/np.sum(mat, axis=0)
    if normalize=='truth':
        mat = mat/np.sum(mat, axis=1)
    labels = list(residue_constants.aatype_to_str_sequence(list(range(20))))
    df_cm = pd.DataFrame(
        mat,
        index=labels,
        columns=labels,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=annot, annot_kws={"size": 16}, fmt='d', ax=ax)
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    #plt.savefig('matrix.svg')
    plt.show()

    
def show_matrices_aa_to_images(mat_list, annot=False, normalize=None):
    m_sqrt = int(np.ceil(np.sqrt(len(mat_list))))
    fig, axes = plt.subplots(m_sqrt, m_sqrt, figsize=(5*len(mat_list), 5))
    sn.set(font_scale=0.8)
    labels = list(residue_constants.aatype_to_str_sequence(list(range(20))))
    #labels = #list(ID_TO_HHBLITS_AA.values())[:20]
    for i, mat in enumerate(mat_list):
        mat = mat[:20,:20]
        if normalize=='pred':
            mat = mat/np.sum(mat, axis=0)
        if normalize=='truth':
            mat = mat/np.sum(mat, axis=1)
        df_cm = pd.DataFrame(
            mat,
            index=labels,
            columns=labels,
        )
        sn.heatmap(df_cm, annot=annot, fmt='d', ax=axes[i//m_sqrt][i%m_sqrt])
        axes[i//m_sqrt][i%m_sqrt].set_title('epoch '+str(i))
        axes[i//m_sqrt][i%m_sqrt].set_xlabel('Prediction')
        axes[i//m_sqrt][i%m_sqrt].set_ylabel('Truth')
    plt.show()


def show_matrix_to_image(mat, annot=False):
    labels = ['positive', 'negative']
    df_cm = pd.DataFrame(
        mat,
        index=labels,
        columns=labels,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    sn.set(font_scale=1.2)
    annot = np.array([['tp', 'fn'], ['fp', 'tn']])
    sn.heatmap(df_cm, annot=annot, annot_kws={"size": 16}, fmt='', ax=ax)
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.show()

    
def show_matrices_to_images(mat_list, annot=False):
    m_sqrt = int(np.ceil(np.sqrt(len(mat_list))))
    fig, axes = plt.subplots(m_sqrt, m_sqrt, figsize=(5*len(mat_list), 5))
    sn.set(font_scale=0.8)
    labels = ['positive', 'negative']
    for i, mat in enumerate(mat_list):
        df_cm = pd.DataFrame(
            mat,
            index=labels,
            columns=labels,
        )
        annot = np.array([['tp', 'fn'], ['fp', 'tn']])
        sn.heatmap(df_cm, annot=annot, fmt='', ax=axes[i//m_sqrt][i%m_sqrt])
        axes[i//m_sqrt][i%m_sqrt].set_title('epoch '+str(i))
        axes[i//m_sqrt][i%m_sqrt].set_xlabel('Prediction')
        axes[i//m_sqrt][i%m_sqrt].set_ylabel('Truth')
    plt.show()


def open_jsons():
    with open(args.file, 'r') as f_json :
        logs = f_json.read()
        split_logs = logs.split('}')[:-1]
        json_dicts = []
        for i in tqdm(range(len(split_logs))):
        #for i in tqdm(range(len(split_logs)-10000,len(split_logs))):
            json_dicts.append(json.loads(split_logs[i]+'}')) # add } because it's removed by the split
    return json_dicts
    

def plot_metric(metric_name, json_dicts, group_by_epoch=False, smooth=1):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    #todo : x axis with batch_idx x epoch_nb
    for i, m_dict in enumerate(json_dicts):
        if metric_name in m_dict.keys() and not np.isnan(m_dict[metric_name]):
            epoch = m_dict['epoch_nb']
            if group_by_epoch:
                metric_data[epoch].append(m_dict[metric_name])
            else:
                metric_data.append(m_dict[metric_name])
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    print('Mean value:', np.mean(metric_data))
    plt.plot(metric_data)
    plt.title(metric_name)
    plt.show()


def x1_plot_confusion_matrix_substitutions(json_dicts, normalize=None, group_by_epoch=False):
    matrix = np.zeros((22,22)).astype('float64')
    matrices_epochs = []
    for i, m_dict in enumerate(json_dicts):
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'seq_anchor' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys():
            for j in range(len(m_dict['seq_truth'])):
                if (m_dict['seq_truth'][j] != m_dict['seq_ancestor'][j]) and m_dict['mask_substitutions_ancestor'] == 0:
                    print('PROBLEME')
                    print(m_dict['seq_truth'][j], m_dict['seq_ancestor'][j])
                    print(m_dict['mask_substitutions_ancestor'])
            seq_truth_sub = [m_dict['seq_truth'][j] for j in range(len(m_dict['seq_truth'])) if m_dict['seq_truth'][j] != m_dict['seq_ancestor'][j]]
            seq_pred_sub = [m_dict['seq_pred'][j] for j in range(len(m_dict['seq_pred'])) if m_dict['seq_truth'][j] != m_dict['seq_ancestor'][j]]
            conf = confusion_matrix(seq_truth_sub, seq_pred_sub, normalize=None, labels=np.arange(0, 22, 1)).astype('float64')
            epoch = m_dict['epoch_nb']
            if group_by_epoch:
                if len(matrices_epochs) <= epoch:
                    matrices_epochs.append(conf)
                else:
                    matrices_epochs[epoch] += conf
            else:
                matrix += conf
    matrix /= len(json_dicts)
    if group_by_epoch:
        show_matrices_aa_to_images(matrices_epochs, normalize=normalize)
    else:
        show_matrix_aa_to_image(matrix, normalize=normalize)

    
def x2_accuracy_mask_substitutions_ancestor(json_dicts):
    accuracies = []
    tot = 0
    gud = 0
    epoch = 1
    for i, m_dict in enumerate(json_dicts):
        if epoch == m_dict['epoch_nb']:
            if tot > 0:
                accuracies.append(gud/tot)
            gud = 0
            tot = 0
            epoch += 1
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'seq_human' in m_dict.keys() and 'seq_mouse' in m_dict.keys() and 'seq_anchor' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys():    # TODO seq_human is not in the log
            for k in range(len(m_dict['seq_truth'])):
                if m_dict['mask_substitutions_ancestor'][k]==1:    # substitution
                    if not int(m_dict['seq_truth'][k]) != int(m_dict['seq_ancestor'][k]):
                        print('STOP, ARRETEZ TOUT, erreur dans le masque des substitutions')
                    tot += 1
                    if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution
                        gud += 1
    accuracies.append(gud/tot)
    plt.plot(accuracies, label="accuracy")
    plt.legend()
    plt.show()
    return
    

json_dicts = open_jsons()

leave = False
keys = list(json_dicts[0].keys())
group_by_epoch = False
smooth=10
while not leave:
    print('\n\nPlease input what you wanna see:')
    for i, k in enumerate(keys):
        print(i, 'for', k)
    print('x1 (nature) for confusion matrix between truth and prediction on substitution positions')
    print('x1b (nature) for confusion matrix between truth and prediction on substitution positions (normalized by truth)')
    print('x1c (nature) for confusion matrix between truth and prediction on substitution positions (normalized by pred)')
    print('x2 (nature) accuracy on mask_substitutions_ancestor')
    print('s to switch between raw data and epoch means')
    print('exit for exit')
    m_input = input()
    if m_input == 's':
        group_by_epoch = not group_by_epoch
        continue
    if m_input == 'exit':
        leave = True
        continue
    try:
        if m_input not in ['x1', 'x1b', 'x1c', 'x2']:
            x = int(m_input)
            if x>len(keys)-1:
                print('Please input a proper number')
                continue
    except:
        print('Please enter a number')
        continue

    if m_input == 'x1':
        print('plot_confusion_matrix_substitutions')
        x1_plot_confusion_matrix_substitutions(json_dicts, normalize=None, group_by_epoch=group_by_epoch)
    elif m_input == 'x1b':
        print('plot_confusion_matrix_substitutions (normalized by truth)')
        x1_plot_confusion_matrix_substitutions(json_dicts, normalize='truth', group_by_epoch=group_by_epoch)
    elif m_input == 'x1c':
        print('plot_confusion_matrix_substitutions (normalized by pred)')
        x1_plot_confusion_matrix_substitutions(json_dicts, normalize='pred', group_by_epoch=group_by_epoch)
    elif m_input == 'x2':
        print('accuracy on mask_substitutions_ancestor')
        x2_accuracy_mask_substitutions_ancestor(json_dicts)
    elif 'seq' in keys[int(m_input)]:
        plot_seq(keys[int(m_input)], json_dicts, group_by_epoch)
    else:
        print('Plotting the metric', keys[int(m_input)])
        plot_metric(keys[int(m_input)], json_dicts, group_by_epoch=group_by_epoch, smooth=smooth)
        
