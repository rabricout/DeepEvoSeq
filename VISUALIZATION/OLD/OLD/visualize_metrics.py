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
    labels = list(ID_TO_HHBLITS_AA.values())[:20]
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
    labels = list(ID_TO_HHBLITS_AA.values())[:20]
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


def plot_f_score(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            if 2*m_dict['tp']+m_dict['fp']+m_dict['fn'] > 0:
                epoch = m_dict['epoch_nb']
                f_score = 2*m_dict['tp'] / (2*m_dict['tp']+m_dict['fp']+m_dict['fn'])
                if group_by_epoch:
                    metric_data[epoch].append(f_score)
                else:
                    metric_data.append(f_score)
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    plt.plot(metric_data)
    plt.title('F-score')
    plt.show()

def plot_f_score_seq_pos(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        seq_pred_pos_a = np.argmax(m_dict['seq_pred_pos'], axis=-1)
        for j in range(len(m_dict['seq_truth'])):
            if m_dict['seq_truth'][j] != 21 and m_dict['seq_anchor'][j] != 21:
                if m_dict['seq_truth'][j]==m_dict['seq_anchor'][j]:    # negative
                    tn += (seq_pred_pos_a[j]==0)
                    fn += (seq_pred_pos_a[j]==1)
                if m_dict['seq_truth'][j]!=m_dict['seq_anchor'][j]:    # positive
                    tp += (seq_pred_pos_a[j]==1)
                    fp += (seq_pred_pos_a[j]==0)
        if 2*tp+fp+fn > 0:
            epoch = m_dict['epoch_nb']
            f_score = 2*tp / (2*tp+fp+fn)
            if group_by_epoch:
                metric_data[epoch].append(f_score)
            else:
                metric_data.append(f_score)
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    plt.plot(metric_data)
    plt.title('F-score (based on sequence prediction)')
    plt.show()
    
def plot_f_score_seq(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        if 'seq_truth' in m_dict.keys():
            for j in range(len(m_dict['seq_truth'])):
                if m_dict['seq_truth'][j] != 21 and m_dict['seq_anchor'][j] != 21 and m_dict['seq_pred'][j] != 21:
                    if m_dict['seq_truth'][j]==m_dict['seq_anchor'][j]:    # negative
                        tn += (m_dict['seq_pred'][j]==m_dict['seq_anchor'][j])
                        fn += (m_dict['seq_pred'][j]!=m_dict['seq_anchor'][j])
                    if m_dict['seq_truth'][j]!=m_dict['seq_anchor'][j]:    # positive
                        tp += (m_dict['seq_pred'][j]!=m_dict['seq_anchor'][j])
                        fp += (m_dict['seq_pred'][j]==m_dict['seq_anchor'][j])
            if 2*tp+fp+fn > 0:
                epoch = m_dict['epoch_nb']
                f_score = 2*tp / (2*tp+fp+fn)
                if group_by_epoch:
                    metric_data[epoch].append(f_score)
                else:
                    metric_data.append(f_score)
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    plt.plot(metric_data)
    plt.title('F-score (based on sequence prediction)')
    plt.show()


def plot_precision_recall(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data_x = [[] for _ in range(epoch_max)] if group_by_epoch else []
    metric_data_y = [[] for _ in range(epoch_max)] if group_by_epoch else []
    colors = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            if m_dict['tp']+m_dict['fp'] > 0 and m_dict['tp']+m_dict['fn'] > 0:
                epoch = m_dict['epoch_nb']
                precision = m_dict['tp'] / (m_dict['tp']+m_dict['fp'])
                recall = m_dict['tp'] / (m_dict['tp']+m_dict['fn'])
                if group_by_epoch:
                    metric_data_x[epoch].append(recall)
                    metric_data_y[epoch].append(precision)
                    colors[epoch].append(epoch)
                else:
                    metric_data_x.append(recall)
                    metric_data_y.append(precision)
                    colors.append(epoch)
    
    if group_by_epoch:
        metric_data_x = [np.mean(d) for d in metric_data_x]
        metric_data_y = [np.mean(d) for d in metric_data_y]
        colors = list(range(epoch_max))
    plt.scatter(metric_data_x, metric_data_y, s=10+200/np.sqrt(len(metric_data_x)), c=colors, cmap='viridis')
    plt.colorbar(label='epochs')
    plt.title('Precision-recall')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.show()
    

def plot_ROC(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    #epoch_max = len(json_dicts) // 20 + 1
    metric_data_x = [[] for _ in range(epoch_max)] if group_by_epoch else []
    metric_data_y = [[] for _ in range(epoch_max)] if group_by_epoch else []
    colors = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            if m_dict['fp']+m_dict['tn'] > 0 and m_dict['tp']+m_dict['fn'] > 0:
                epoch = m_dict['epoch_nb']
                fpr = m_dict['fp'] / (m_dict['fp']+m_dict['tn'])
                tpr = m_dict['tp'] / (m_dict['tp']+m_dict['fn'])
                if group_by_epoch:
                    metric_data_x[epoch].append(fpr)  # epoch instead of i//10
                    metric_data_y[epoch].append(tpr)
                    colors[epoch].append(epoch)
                else:
                    metric_data_x.append(fpr)
                    metric_data_y.append(tpr)
                    colors.append(epoch)
    if group_by_epoch:
        metric_data_x = [np.mean(d) for d in metric_data_x]
        metric_data_y = [np.mean(d) for d in metric_data_y]
        colors = list(range(epoch_max))
    plt.scatter(metric_data_x, metric_data_y, s=10+200/np.sqrt(len(metric_data_x)), c=colors, cmap='viridis')
    plt.colorbar(label='training steps')
    plt.title('ROC')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.show()
    

def plot_seq(metric_name, json_dicts, epoch_mean):
    return

def accuracy_by_category(json_dicts, outgroup_species='mouse'):
    outgroup_id = 'seq_'+outgroup_species
    accuracies_target = []
    accuracies_anchor = []
    accuracies = []
    epoch = 1
    gud_t = 0
    tot_t = 0
    gud_a = 0
    tot_a = 0
    gud = 0
    tot = 0
    for i, m_dict in enumerate(json_dicts):
        if epoch == m_dict['epoch_nb']:
            if tot_t > 0:
                accuracies_target.append(gud_t/tot_t)
            if tot_a > 0:
                accuracies_anchor.append(gud_a/tot_a)
            if tot > 0:
                accuracies.append(gud/tot)
            gud_t = 0
            tot_t = 0
            gud_a = 0
            tot_a = 0
            gud = 0
            tot = 0
            epoch += 1
            
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and outgroup_id in m_dict.keys() and 'seq_anchor' in m_dict.keys():    # TODO seq_human is not in the log
            if False:
                print(residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_truth']).astype(int)))
                print()
                print(residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_anchor']).astype(int)))
                print()
                print(residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_human']).astype(int)))
                print()
                print(residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_mouse']).astype(int)))
                print()
                #print(residue_constants.aatype_to_str_sequence(np.array(m_dict[outgroup_id]).astype(int)))
                #print()
                #print(residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_pred']).astype(int)))
                #print()
                input()
            for k in range(len(m_dict['seq_truth'])):
                if int(m_dict['seq_truth'][k]) != int(m_dict['seq_anchor'][k]):    # substitution
                    tot += 1
                    if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):
                        gud += 1
                    if int(m_dict[outgroup_id][k]) == int(m_dict['seq_anchor'][k]):    # substitution on target branch
                        tot_t += 1
                        if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution
                            gud_t += 1
                    else:    # substitution on anchor branch
                        #print(k, int(m_dict['seq_truth'][k]), int(m_dict['seq_anchor'][k]), int(m_dict['seq_mouse'][k]), int(m_dict['seq_pred'][k]))
                        tot_a += 1
                        if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution
                            gud_a += 1
            #input()
    #print('Target', gud_t, tot_t, gud_t/tot_t)
    #print('Anchor', gud_a, tot_a, gud_a/tot_a)
    accuracies_target.append(gud_t/tot_t)
    accuracies_anchor.append(gud_a/tot_a)
    accuracies.append(gud/tot)
    plt.plot(accuracies_target, label="target")
    plt.plot(accuracies_anchor, label="anchor")
    plt.plot(accuracies, label="tot")
    plt.legend()
    plt.show()
    return


def accuracy_by_category_biancestor(json_dicts):
    accuracies_target = []
    accuracies_anchor = []
    accuracies_other1  = []
    accuracies_other2  = []
    accuracies_other3  = []
    accuracies = []
    epoch = 1
    gud_t = 0
    tot_t = 0
    gud_a = 0
    tot_a = 0
    gud_o1 = 0
    tot_o1 = 0
    gud_o2 = 0
    tot_o2 = 0
    gud_o3 = 0
    tot_o3 = 0
    count_t = 0
    count_a = 0
    count_o1 = 0
    count_o2 = 0
    count_o3 = 0
    gud = 0
    tot = 0
    for i, m_dict in enumerate(json_dicts):
        if epoch == m_dict['epoch_nb']:
            if tot_t > 0:
                accuracies_target.append(gud_t/tot_t)
            if tot_a > 0:
                accuracies_anchor.append(gud_a/tot_a)
            if tot_o1 > 0:
                accuracies_other1.append(gud_o1/tot_o1)
            if tot_o2 > 0:
                accuracies_other2.append(gud_o2/tot_o2)
            if tot_o3 > 0:
                accuracies_other3.append(gud_o3/tot_o3)
            if tot > 0:
                accuracies.append(gud/tot)
            gud_t = 0
            tot_t = 0
            gud_a = 0
            tot_a = 0
            gud_o1 = 0
            tot_o1 = 0
            gud_o2 = 0
            tot_o2 = 0
            gud_o3 = 0
            tot_o3 = 0
            gud = 0
            tot = 0
            epoch += 1
            
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'seq_human' in m_dict.keys() and 'seq_mouse' in m_dict.keys() and 'seq_anchor' in m_dict.keys():    # TODO seq_human is not in the log
            for k in range(len(m_dict['seq_truth'])):
                if int(m_dict['seq_truth'][k]) != int(m_dict['seq_anchor'][k]):    # substitution
                #if m_dict['mask_substitutions_ancestor'][k]==1:    # substitution
                    tot += 1
                    if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):
                        gud += 1
                    if int(m_dict['seq_human'][k]) == int(m_dict['seq_mouse'][k]) and int(m_dict['seq_human'][k]) != int(m_dict['seq_anchor'][k]):    # subsitution in anchor branch
                        tot_a += 1
                        count_a += 1
                        if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution
                            gud_a += 1
                    elif int(m_dict['seq_human'][k]) == int(m_dict['seq_mouse'][k]) and int(m_dict['seq_human'][k]) == int(m_dict['seq_anchor'][k]):    # subsitution in target branch
                        tot_t += 1
                        count_t += 1
                        if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution
                            gud_t += 1
                    else:    # other cases
                        if int(m_dict['seq_human'][k]) == int(m_dict['seq_anchor'][k]):
                            tot_o1 += 1
                            count_o1 += 1
                            if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution
                                gud_o1 += 1
                        elif int(m_dict['seq_mouse'][k]) == int(m_dict['seq_anchor'][k]):
                            tot_o2 += 1
                            count_o2 += 1
                            if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution
                                gud_o2 += 1
                        else:
                            tot_o3 += 1
                            count_o3 += 1
                            if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution
                                gud_o3 += 1
                            
                                
            #input()
    #print('Target', gud_t, tot_t, gud_t/tot_t)
    #print('Anchor', gud_a, tot_a, gud_a/tot_a)
    print('target', count_t/(count_t+count_o1+count_o2+count_o3+count_a))
    print('anchor', count_a/(count_t+count_o1+count_o2+count_o3+count_a))
    print('other (human=anchor)', count_o1/(count_t+count_o1+count_o2+count_o3+count_a))
    print('other (human=mouse)', count_o2/(count_t+count_o1+count_o2+count_o3+count_a))
    print('other (other)', count_o3/(count_t+count_o1+count_o2+count_o3+count_a))
    accuracies_other1.append(gud_o1/tot_o1)
    accuracies_other2.append(gud_o2/tot_o2)
    accuracies_other3.append(gud_o3/tot_o3)
    accuracies_target.append(gud_t/tot_t)
    accuracies_anchor.append(gud_a/tot_a)
    accuracies.append(gud/tot)
    plt.plot(accuracies_target, label="target")
    plt.plot(accuracies_anchor, label="anchor")
    plt.plot(accuracies_other1, label="other (human=anchor)")
    plt.plot(accuracies_other2, label="other (human=mouse)")
    plt.plot(accuracies_other3, label="other (other)")
    plt.plot(accuracies, label="tot")
    plt.legend()
    plt.show()
    return


def accuracy_by_category_biancestor_b(json_dicts):
    accuracies_anchor = []
    accuracies_other  = []
    accuracies = []
    epoch = 1
    gud_a = 0
    tot_a = 0
    gud_o = 0
    tot_o = 0
    count_a = 0
    count_o = 0
    for i, m_dict in enumerate(json_dicts):
        if epoch == m_dict['epoch_nb']:
            if tot_a > 0:
                accuracies_anchor.append(gud_a/tot_a)
            if tot_o > 0:
                accuracies_other.append(gud_o/tot_o)
            gud_a = 0
            tot_a = 0
            gud_o = 0
            tot_o = 0
            epoch += 1
            
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'seq_human' in m_dict.keys() and 'seq_mouse' in m_dict.keys() and 'seq_anchor' in m_dict.keys():    # TODO seq_human is not in the log
            for k in range(len(m_dict['seq_truth'])):
                if int(m_dict['seq_truth'][k]) != int(m_dict['seq_anchor'][k]):    # substitution
                    if int(m_dict['seq_human'][k]) == int(m_dict['seq_mouse'][k]) and int(m_dict['seq_human'][k]) != int(m_dict['seq_anchor'][k]):    # subsitution in anchor branch
                        tot_a += 1
                        count_a += 1
                        if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution
                            gud_a += 1
                    else:    # other cases
                        tot_o += 1
                        count_o += 1
                        if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution
                            gud_o += 1
            #input()
    #print('Target', gud_t, tot_t, gud_t/tot_t)
    #print('Anchor', gud_a, tot_a, gud_a/tot_a)
    print('anchor', count_a/(count_a+count_o))
    print('others', count_o/(count_a+count_o))
    accuracies_other.append(gud_o/tot_o)
    accuracies_anchor.append(gud_a/tot_a)
    plt.plot(accuracies_anchor, label="anchor")
    plt.plot(accuracies_other,  label="other")
    plt.legend()
    plt.show()
    return

    
def plot_confusion_matrix(json_dicts, normalize=None, group_by_epoch=False):
    matrix = np.zeros((22,22))
    matrices_epochs = []
    for i, m_dict in enumerate(json_dicts):
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys():
            conf = confusion_matrix(m_dict['seq_truth'], m_dict['seq_pred'], normalize=None, labels=np.arange(0, 22, 1))
            epoch = m_dict['epoch_nb']
            if group_by_epoch:
                if len(matrices_epochs) <= epoch:
                    matrices_epochs.append(conf)
                else:
                    matrices_epochs[epoch] += conf
            else:
                matrix += conf
    if group_by_epoch:
        show_matrices_aa_to_images(matrices_epochs, normalize=normalize)
    else:
        matrix /= len(json_dicts)
        show_matrix_aa_to_image(matrix, normalize=normalize)
        

def plot_confusion_matrix_substitutions(json_dicts, normalize=None, group_by_epoch=False):
    matrix = np.zeros((22,22)).astype('float64')
    matrices_epochs = []
    for i, m_dict in enumerate(json_dicts):
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys():
            seq_truth_sub = [m_dict['seq_truth'][i] for i in range(len(m_dict['seq_truth'])) if m_dict['seq_truth'][i] != m_dict['seq_anchor'][i]]
            seq_pred_sub = [m_dict['seq_pred'][i] for i in range(len(m_dict['seq_pred'])) if m_dict['seq_truth'][i] != m_dict['seq_anchor'][i]]
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

        
def plot_pos_confusion_matrix(json_dicts, group_by_epoch=False):
    matrix = np.zeros((2, 2))
    matrices_epochs = []
    for i, m_dict in enumerate(json_dicts):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            epoch = m_dict['epoch_nb']
            conf = np.zeros((2, 2))
            conf[0,0] += m_dict['tp']
            conf[0,1] += m_dict['fp']
            conf[1,0] += m_dict['fn']
            conf[1,1] += m_dict['tn']
            if group_by_epoch:
                if len(matrices_epochs) <= epoch:
                    matrices_epochs.append(conf)
                else:
                    matrices_epochs[epoch] += conf
            else:
                matrix += conf
    if group_by_epoch:
        show_matrices_to_images(matrices_epochs)
    else:
        matrix /= len(json_dicts)
        show_matrix_to_image(matrix)


def plot_substitutions_positions(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    subst_pos_truth = [np.zeros((100)) for _ in range(epoch_max)] if group_by_epoch else np.zeros((100))
    subst_pos_pred = [np.zeros((100)) for _ in range(epoch_max)] if group_by_epoch else np.zeros((100))
    tot_pos = [np.zeros((100)) for _ in range(epoch_max)] if group_by_epoch else np.zeros((100))
    for m_dict in json_dicts:
        length = m_dict['seq_length']
        epoch = m_dict['epoch_nb']
        for i, index in enumerate(m_dict['residue_index'][0]):
            local_bin = int(index/length*100)
            if group_by_epoch:
                subst_pos_truth[epoch][local_bin] += (m_dict['seq_truth'][i] != m_dict['seq_anchor'][i])
                subst_pos_pred[epoch][local_bin]  += (m_dict['seq_pred'][i]  != m_dict['seq_anchor'][i])
                tot_pos[epoch][local_bin] += 1
            elif epoch_max==1 or m_dict['epoch_nb'] > 0:
                subst_pos_truth[local_bin] += (m_dict['seq_truth'][i] != m_dict['seq_anchor'][i])
                subst_pos_pred[local_bin]  += (m_dict['seq_pred'][i]  != m_dict['seq_anchor'][i])
                tot_pos[local_bin] += 1
    if group_by_epoch:
        metric_data_truth = [subst_pos_truth[i] / tot_pos[i] for i in range(epoch_max)]
        metric_data_pred  = [subst_pos_pred[i]  / tot_pos[i] for i in range(epoch_max)]
        for i in range(epoch_max):
            plt.plot(metric_data_truth[i], label='truth '+str(i))
            plt.plot(metric_data_pred[i], label='predictions '+str(i), linestyle='dashed')
    else:
        metric_data_truth = subst_pos_truth / tot_pos
        metric_data_pred = subst_pos_pred / tot_pos
        plt.plot(metric_data_truth, label='truth')
        plt.plot(metric_data_pred, label='predictions', linestyle='dashed')
    plt.legend(loc='upper left')
    plt.title('Substitutions according to position')
    plt.show()


def plot_subst_rate_diff(json_dicts):
    subst_diff_pred = []
    subst_diff_proxy = []
    subst_pred = []
    subst_truth = []
    subst_human = []
    subst_proxy = []
    for m_dict in json_dicts:
        if 'substitution_rate_truth_CM' in m_dict.keys() and 'substitution_rate_pred_CM' in m_dict.keys():
            subst_diff_pred.append(np.abs(m_dict['substitution_rate_truth_CM'] - m_dict['substitution_rate_pred_CM']))
            subst_diff_proxy.append(np.abs(m_dict['substitution_rate_truth_CM'] - m_dict['substitution_rate_truth_HM']/0.11*0.083))
            if m_dict['substitution_rate_truth_CM'] > 0.5:
                print('seqs')
                s_t = residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_truth']).astype(int))
                s_a = residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_anchor']).astype(int))
                print("m_dict['seq_truth']", s_t)
                print("m_dict['seq_anchor']", s_a)
            subst_pred.append(m_dict['substitution_rate_pred_CM'])
            subst_truth.append(m_dict['substitution_rate_truth_CM'])
            subst_human.append(m_dict['substitution_rate_truth_HM'])
            subst_proxy.append(m_dict['substitution_rate_truth_HM']/0.11*0.083)
    plt.plot(subst_diff_pred, label='diff (abs) between pred and truth', linestyle=':', c='blue')
    plt.plot(subst_diff_proxy, label='diff (abs) between proxy and truth', linestyle=':', c='green')
    plt.plot(subst_truth, label='truth subst rate CM (beaver/squirrel)', c='red')
    plt.plot(subst_pred, label='predicted subst rate CM (beaver/squirrel)', c='blue')
    plt.plot(subst_proxy, label='proxy of subst rate CM (beaver/squirrel)', c='green')
    plt.plot(subst_human, label='truth subst rate HM (beaver/squirrel)', linestyle='--', c='black')
    plt.legend(loc='upper left')
    plt.title('Distance between predictions and truth of substitution rate')
    plt.show()

    
def plot_subst_rates(json_dicts):
    subst_rate_CM = []
    subst_rate_HM = []
    subst_rate_CM_noIndels = []
    subst_rate_HM_noIndels = []
    subst_rate_CM_pred = []
    for m_dict in json_dicts:
        if 'substitution_rate_truth_CM' in m_dict.keys() and 'substitution_rate_pred_CM' in m_dict.keys():
            subst_rate_CM.append(m_dict['substitution_rate_truth_CM'])
            subst_rate_HM.append(m_dict['substitution_rate_truth_HM'])
            subst_rate_CM_noIndels.append(m_dict['substitution_rate_truth_CM_noIndels'])
            subst_rate_HM_noIndels.append(m_dict['substitution_rate_truth_HM_noIndels'])
            subst_rate_CM_pred.append(m_dict['substitution_rate_pred_CM'])
    print(sum(subst_rate_CM_noIndels)/len(subst_rate_CM_noIndels))
    plt.plot(subst_rate_CM, label='subst_rate_CM')
    plt.plot(subst_rate_HM, label='subst_rate_HM')
    plt.plot(subst_rate_CM_noIndels, label='subst_rate_CM_noIndels')
    plt.plot(subst_rate_HM_noIndels, label='subst_rate_HM_noIndels')
    plt.plot(subst_rate_CM_pred, label='subst_rate_pred_CM')
    plt.legend(loc='upper left')
    plt.title('substitution_rates')
    plt.show()


def pos_subst_rate(json_dicts):
    subst_rate_CM = []
    subst_rate_CM_pos_pred = []
    epoch = 1
    subst = 0
    tot = 0
    local_true_subst_rate = []
    for m_dict in json_dicts:
        if 'substitution_rate_truth_CM' in m_dict.keys() and 'tp' in m_dict.keys():
            if epoch == m_dict['epoch_nb']:
                subst_rate_CM_pos_pred.append(subst/tot)
                subst_rate_CM.append(np.mean(np.array(local_true_subst_rate)))
                subst = 0
                tot = 0
                epoch += 1
            local_true_subst_rate.append(m_dict['substitution_rate_truth_CM'])
            subst += m_dict['tp']+m_dict['fp']
            tot += m_dict['tp']+m_dict['fp']+m_dict['tn']+m_dict['fn']
    subst_rate_CM_pos_pred.append(subst/tot)
    subst_rate_CM.append(np.mean(np.array(local_true_subst_rate)))
    plt.plot(subst_rate_CM, label='subst_rate_CM')
    plt.plot(subst_rate_CM_pos_pred, label='subst_rate_pred_pos_CM')
    plt.legend(loc='upper left')
    plt.title('substitution_rates')
    plt.show()
    
def print_correlations(json_dicts):
    subst_CM = []
    subst_HM = []
    subst_CM_pred = []
    last_1000_dicts = json_dicts[-1000:] if len(json_dicts) > 1000 else json_dicts
    for m_dict in last_1000_dicts:
        if 'substitution_rate_truth_CM' in m_dict.keys() and 'substitution_rate_pred_CM' in m_dict.keys():
            subst_CM.append(m_dict['substitution_rate_truth_CM'])
            subst_HM.append(m_dict['substitution_rate_truth_HM'])
            subst_CM_pred.append(m_dict['substitution_rate_pred_CM'])
    subst_CM = np.array(subst_CM)
    subst_HM = np.array(subst_HM)
    subst_CM_pred = np.array(subst_CM_pred)
    #pearson_correlation_truth = scipy.stats.pearsonr(subst_CM, subst_HM)
    #pearson_correlation_pred  = scipy.stats.pearsonr(subst_CM_pred, subst_HM)
    pearson_correlation_CM_pred = scipy.stats.pearsonr(subst_CM_pred, subst_CM)
    # defining proxy
    mean_CM = np.linalg.norm(subst_CM, ord=1) / len(subst_CM)
    mean_HM = np.linalg.norm(subst_HM, ord=1) / len(subst_HM)
    #print(mean_HM, 0.11)
    #print(mean_CM, 0.083)
    subst_CM_proxy = subst_HM / 0.11*0.083 #mean_HM * mean_CM
    pearson_correlation_CM_proxy = scipy.stats.pearsonr(subst_CM_proxy, subst_CM)
    
    diff_pred_truth_CM_L1  = np.linalg.norm((subst_CM-subst_CM_pred), ord=1) / len(subst_CM-subst_CM_pred)
    diff_proxy_truth_CM_L1 = np.linalg.norm((subst_CM-subst_CM_proxy), ord=1) / len(subst_CM-subst_CM_proxy)
    diff_pred_truth_CM_L2  = np.linalg.norm((subst_CM-subst_CM_pred), ord=2) / len(subst_CM-subst_CM_pred)
    diff_proxy_truth_CM_L2 = np.linalg.norm((subst_CM-subst_CM_proxy), ord=2) / len(subst_CM-subst_CM_proxy)

    #print('pearson_correlation between CM and HM', pearson_correlation_truth)
    #print('pearson_correlation between CM predicted and HM', pearson_correlation_pred)
    print('pearson_correlation between CM predicted and CM truth', pearson_correlation_CM_pred)
    print('pearson_correlation between CM proxy and CM truth', pearson_correlation_CM_proxy)
    print('mean distance (L1 norm) between CM pred and CM truth', diff_pred_truth_CM_L1)
    print('mean distance (L1 norm) between CM proxy and CM truth', diff_proxy_truth_CM_L1)
    print('mean distance (L2 norm) between CM pred and CM truth', diff_pred_truth_CM_L2)
    print('mean distance (L2 norm) between CM proxy and CM truth', diff_proxy_truth_CM_L2)
    input()


def compare_subst_rates(json_dicts):
    subst_truth = []
    subst_pred = []
    subst_pred_seq = []
    subst_pred_pos_softmax = []
    subst_pred_pos_sharpsoftmax = []
    subst_pred_pos_argmax = []
    for m_dict in json_dicts:
        subst_truth.append(m_dict['substitution_rate_truth_CM'])
        subst_pred.append(m_dict['substitution_rate_pred_CM'])
        subst_pos_pred = m_dict['seq_pred_pos']
        eps=0.1
        subst_pos_sm  = softmax(subst_pos_pred, axis=1)
        subst_pos_ssm = softmax(np.array(subst_pos_pred)/eps, axis=1)
        subst_pos_am  = np.argmax(subst_pos_pred, axis=1)
        subst_pred_pos_softmax.append(sum(subst_pos_sm[:,1]) / len(subst_pos_sm))
        subst_pred_pos_sharpsoftmax.append(sum(subst_pos_ssm[:,1]) / len(subst_pos_ssm))
        subst_pred_pos_argmax.append (sum(subst_pos_am) / len(subst_pos_am))
        seq_anchor = m_dict['seq_anchor']
        seq_pred   = m_dict['seq_pred']
        subst = 0
        tot = 0
        for i in range(len(seq_pred)):
            if seq_anchor[i] != 21 and seq_pred[i] != 21:
                tot += 1
                subst += (seq_anchor[i] != seq_pred[i])
        subst_pred_seq.append(subst/tot)
    plt.plot(subst_truth, label='subst_truth')
    plt.plot(subst_pred, label='subst_pred_branch')
    plt.plot(subst_pred_seq, label='subst_pred_seq')
    plt.plot(subst_pred_pos_softmax, label='subst_pred_pos_softmax')
    plt.plot(subst_pred_pos_sharpsoftmax, label='subst_pred_pos_sharpsoftmax (eps='+str(eps)+')')
    plt.plot(subst_pred_pos_argmax,  label='subst_pred_pos_argmax')
    plt.legend(loc='upper left')
    plt.show()
            
        
def subst_rates_scatter(json_dicts):
    subst_rate = []
    subst_rate_pred = []
    subst_HM = []
    for m_dict in json_dicts:
        if 'substitution_rate_truth_CM' in m_dict.keys() and 'substitution_rate_pred_CM' in m_dict.keys():
            subst_rate.append(m_dict['substitution_rate_truth_CM'])
            subst_rate_pred.append(m_dict['substitution_rate_pred_CM'])
            subst_HM.append(m_dict['substitution_rate_truth_HM'])
    subst_rate_proxy = np.array(subst_HM) / 0.11*0.083 #mean_HM * mean_CM
    
    plt.scatter(subst_rate, subst_rate_pred, label='prediction with neural network', s=5)
    plt.scatter(subst_rate, subst_rate_proxy, label='prediction with proxy', s=5)
    plt.plot([0,1],[0,1], color='black')
    plt.legend(loc='upper left')
    plt.xlabel('true substitution rate')
    plt.ylabel('predicted substitution rate')
    plt.title('scatter plot comparing neural network predictions with proxy')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.show()


def phi(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            N = m_dict['tn'] + m_dict['tp'] + m_dict['fn'] + m_dict['fp']
            S = (m_dict['tp']+m_dict['fn'])/N
            P = (m_dict['tp']+m_dict['fp'])/N
            if np.sqrt(P*S*(1-S)*(1-P)) > 0:
                epoch = m_dict['epoch_nb']
                phi = (m_dict['tp']/N-S*P) / np.sqrt(P*S*(1-S)*(1-P))
                if group_by_epoch:
                    metric_data[epoch].append(phi)
                else:
                    metric_data.append(phi)
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    plt.plot(metric_data)
    plt.title('phi')
    plt.show()

def youdenJ(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            if m_dict['tp']+m_dict['fn'] > 0 and m_dict['tn']+m_dict['fp'] > 0:
                epoch = m_dict['epoch_nb']
                j = m_dict['tp']/(m_dict['tp']+m_dict['fn']) + m_dict['tn']/(m_dict['tn']+m_dict['fp']) - 1
                if group_by_epoch:
                    metric_data[epoch].append(j)
                else:
                    metric_data.append(j)
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    plt.plot(metric_data)
    plt.title('Youden J')
    plt.show()

def cohenKappa(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            num = 2*(m_dict['tp']*m_dict['tn']-m_dict['fn']*m_dict['fp'])
            denom = (m_dict['tp']+m_dict['fp']) * (m_dict['fp']+m_dict['tn']) + (m_dict['tp']+m_dict['fn']) * (m_dict['fn']+m_dict['tn'])
            if denom > 0:
                epoch = m_dict['epoch_nb']
                kappa = num / denom
                if group_by_epoch:
                    metric_data[epoch].append(kappa)
                else:
                    metric_data.append(kappa)
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    plt.plot(metric_data)
    plt.title('Cohen kappa')
    plt.show()


    
def accuracy_mask_substitutions_ancestor(json_dicts):
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
normalize = None #None #'truth' #'pred'
smooth=10
while not leave:
    print('\n\nPlease input what you wanna see:')
    for i, k in enumerate(keys):
        print(i, 'for', k)
    print('x1 (nature) for confusion matrix between truth and prediction')
    print('x2 (nature) for confusion matrix between truth and prediction on substitution positions')
    print('x3 for position of substitutions (excluding epoch 0)')
    print('x4 for position confusion matrix')
    print('x5 (position) for F-score (position head)')
    print('x5b(nature) for F-score (sequence head)')
    print('x6 (position) for precision-recall curve')
    print('x7 (position) for ROC curve')
    print('x8 (substRate) for difference between substitution rate truth and predicted')
    print('x9 (substRate) for correlation between subst_rate CM (castor/marmotte) and subst_rate HM (humain/marmotte)')
    print('x10 (substRate) for subst_rates')
    print('x11 for comparison between true subst rate and subst rate in seq predictions')
    print('x12 (substRate) for subst rates in cloud (pred vs truth)')
    print('x13 (nature) for accuracy by category (based on human as outgroup)')
    print('x14 (nature) for accuracy by category (based on mouse as outgroup)')
    print('x15 (nature) for accuracy by category (using human AND mouse)')
    print('x15b (nature) for accuracy by category (grouped anchor vs all others)')
    print('x16 (positions) for susbtitution rate comparison between truth and position predictions')
    print('x17 (positions) phi coefficient')
    print('x18 (positions) Youden J statistic')
    print('x19 (positions) Cohen kappa')
    print('x20 (nature) accuracy on mask_subtitutions_ancestor')
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
        if m_input not in ['x1', 'x2', 'x3', 'x4', 'x5', 'x5b', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x15b', 'x16', 'x17', 'x18', 'x19', 'x20']:
            x = int(m_input)
            if x>len(keys)-1:
                print('Please input a proper number')
                continue
    except:
        print('Please enter a number')
        continue

    if m_input == 'x1':
        print('plot_confusion_matrix')
        plot_confusion_matrix(json_dicts, normalize=normalize, group_by_epoch=group_by_epoch)
    elif m_input == 'x2':
        print('plot_confusion_matrix_substitutions')
        plot_confusion_matrix_substitutions(json_dicts, normalize=normalize, group_by_epoch=group_by_epoch)
    elif m_input == 'x3':
        print('plot_positions_of_substitutions')
        plot_substitutions_positions(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x4':
        print('plot_confusion_matrix for position')
        plot_pos_confusion_matrix(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x5':
        print('plot_F_score')
        #plot_f_score_seq_pos(json_dicts, group_by_epoch=group_by_epoch)
        plot_f_score(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x5b':
        print('plot_F_score')
        plot_f_score_seq(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x6':
        print('plot_precision_recall_curve')
        plot_precision_recall(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x7':
        print('plot_ROC_curve')
        plot_ROC(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x8':
        print('plot_difference of substitution rate between truth and prediction')
        plot_subst_rate_diff(json_dicts)
    elif m_input == 'x9':
        print('correlations of substitution rates (proxy for branch length for the 1000 last data)')
        print_correlations(json_dicts)
    elif m_input == 'x10':
        print('substitution rates (proxy for branch length)')
        plot_subst_rates(json_dicts)
    elif m_input == 'x11':
        print('subst rates comparison (between truth and seq pred)')
        compare_subst_rates(json_dicts)
    elif m_input == 'x12':
        print('subst rates cloud (truth vs pred)')
        subst_rates_scatter(json_dicts)
    elif m_input == 'x13':
        print('accuracy by category (human outgroup)')
        accuracy_by_category(json_dicts, 'human')
    elif m_input == 'x14':
        print('accuracy by category (mouse outgroup)')
        accuracy_by_category(json_dicts, 'mouse')
    elif m_input == 'x15':
        print('accuracy by category (mouse AND human outgroups)')
        accuracy_by_category_biancestor(json_dicts)
    elif m_input == 'x15b':
        print('accuracy by category (mouse AND human outgroups)')
        accuracy_by_category_biancestor_b(json_dicts)
    elif m_input == 'x16':
        print('subst rate for the positions')
        pos_subst_rate(json_dicts)
    elif m_input == 'x17':
        print('(positions) phi coef')
        phi(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x18':
        print('(positions) Youden J statitic')
        youdenJ(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x19':
        print('(positions) Cohen Kappa')
        cohenKappa(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x20':
        print('accuracy on mask_subtitutions_ancestor')
        accuracy_mask_substitutions_ancestor(json_dicts)
    elif 'seq' in keys[int(m_input)]:
        plot_seq(keys[int(m_input)], json_dicts, group_by_epoch)
    else:
        print('Plotting the metric', keys[int(m_input)])
        plot_metric(keys[int(m_input)], json_dicts, group_by_epoch=group_by_epoch, smooth=smooth)
        
