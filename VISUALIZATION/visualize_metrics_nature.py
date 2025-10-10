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
matplotlib.rcParams.update({'font.size': 12})

import scipy.stats as st
import residue_constants as residue_constants
import utils as utils

from PIL import Image
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from collections import Counter

#matplotlib.use('TkAgg')


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()


def show_matrix_aa_to_image(mat, annot=False, normalize=None):
    mat = mat[:20,:20]
    if normalize=='pred':
        row_sums = mat.sum(axis=0)
        mat = mat / row_sums[:, np.newaxis]
    if normalize=='truth':
        row_sums = mat.sum(axis=1)
        mat = mat / row_sums[:, np.newaxis]
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
    #with open('matrix.npy', 'wb') as f:
    #    np.save(f, mat)
    print(labels)
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
    #else:
    #    metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    print('Mean value:', np.mean(metric_data))
    plt.plot(metric_data)
    plt.title(metric_name)
    plt.show()


def x1_plot_confusion_matrix_substitutions(json_dicts, normalize=None, group_by_epoch=False):
    matrix = np.zeros((22,22)).astype('float64')
    matrices_epochs = []
    for i, m_dict in enumerate(json_dicts[-2000:]):
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

    
def x2_accuracy_mask_substitutions_ancestor(json_dicts, group_by_epoch=False, smooth=0):
    accuracies = []
    tot_all = 0
    gud_all = 0
    tot = 0
    gud = 0
    epoch = 1
    for i, m_dict in enumerate(json_dicts[-50000:]):
        if group_by_epoch:
            if epoch == m_dict['epoch_nb']:
                if tot > 0:
                    accuracies.append(gud/tot)
                gud = 0
                tot = 0
                epoch += 1
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys():    # TODO seq_human is not in the log
            for k in range(len(m_dict['seq_truth'])):
                if m_dict['mask_substitutions_ancestor'][k]==1:    # substitution
                    tot += 1
                    tot_all += 1
                    if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution)
                        gud += 1
                        gud_all += 1
        if not group_by_epoch:
            if tot > 0:
                accuracies.append(gud/tot)
            gud = 0
            tot = 0
    if group_by_epoch:
        accuracies.append(gud/tot)
    if smooth > 0:
        accuracies = np.convolve(accuracies, np.ones(smooth)/smooth, mode='valid')
    acc = np.array(accuracies)
    interval = st.t.interval(confidence=0.95, df=len(acc)-1, loc=np.mean(acc), scale=st.sem(acc))
    print(interval)
    print('Mean of accuracies', np.mean(np.array(accuracies)))
    print('Mean accuracy', gud_all/tot_all)
    plt.plot(accuracies, label="accuracy")
    plt.legend()
    plt.show()
    return

def x3_accuracy_most_least(json_dicts, smooth=0):
    accuracies = []
    subst_rates = []
    tot = []
    gud = []
    for i, m_dict in enumerate(json_dicts): #[-2000:]):
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys():    # TODO seq_human is not in the log
            loc_tot = 0
            loc_gud = 0
            for k in range(len(m_dict['seq_truth'])):
                if m_dict['mask_substitutions_ancestor'][k]==1:    # substitution
                    loc_tot += 1
                    if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution)
                        loc_gud += 1
            if loc_tot > 0:
                subst_rates.append(loc_tot/len(m_dict['mask_substitutions_ancestor']))
                accuracies.append(loc_gud/loc_tot)
                tot.append(loc_tot)
                gud.append(loc_gud)

    zipped = zip(subst_rates, accuracies, tot, gud)
    subst_rates, accuracies, tot, gud = zip(*sorted(zipped))

    pc = int(len(subst_rates)*0.5)

    accuracies_most = accuracies[-pc:]
    accuracies_least = accuracies[:pc]
    tot_most = sum(tot[-pc:])
    tot_least = sum(tot[:pc])
    gud_most = sum(gud[-pc:])
    gud_least = sum(gud[:pc])
        
    #acc = np.array(accuracies)
    #interval = st.t.interval(confidence=0.95, df=len(acc)-1, loc=np.mean(acc), scale=st.sem(acc))
    #print(interval)
    accuracies_least_f = [a for a in accuracies_least if a >=0]
    print('Mean of accuracies most', np.mean(np.array(accuracies_most)))
    print('Mean of accuracies least', np.mean(np.array(accuracies_least_f)))
    print('Mean accuracy most', gud_most, tot_most, gud_most/tot_most)
    print('Mean accuracy least', gud_least, tot_least, gud_least/tot_least)

    if smooth > 0:
        accuracies = np.convolve(accuracies, np.ones(smooth)/smooth, mode='valid')
        subst_rates = np.convolve(subst_rates, np.ones(smooth)/smooth, mode='valid')
    plt.plot(subst_rates, accuracies, label="accuracy according to subst rate (smoothed with window of "+str(smooth)+")")
    plt.xscale('log')
    plt.ylim(0,1)
    plt.xlim(0,0.5)
    plt.legend()
    plt.show()
    plt.savefig('TEST.pdf')
    return

def x4_accuracy_position(json_dicts):
    subst_tot = np.zeros(100)
    subst_gud = np.zeros(100)
    for i, m_dict in enumerate(json_dicts[-10000:]):
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys(): # and 'residue_index' in m_dict.keys():
            for k in range(len(m_dict['seq_truth'])):
                #n_bin = int(m_dict['residue_index'][k]/m_dict['seq_length']*100)
                n_bin = int(k/len(m_dict['seq_truth'])*100)
                if m_dict['mask_substitutions_ancestor'][k]==1:    # substitution
                    subst_tot[n_bin] += 1
                    if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution)
                        subst_gud[n_bin] += 1
    subst = subst_gud/subst_tot
    print(subst)
    plt.ylim(0.18,0.4)
    plt.plot(subst, label="accuracy according to the position")
    plt.legend()
    plt.show()
    return

def x5_accuracy_conservation(json_dicts):
    nb_aa = [str(i) for i in range(20)]
    keys = ['gaps'] + nb_aa
    print(keys)
    subst_tot = {k:0 for k in keys}
    subst_gud = {k:0 for k in keys}
    for i, m_dict in enumerate(json_dicts[-10000:]):
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys() and 'msa' in m_dict.keys():
            human_str = residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_human']).astype(int))
            mouse_str = residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_mouse']).astype(int))
            anchor_str = residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_anchor']).astype(int))
            for k in range(len(m_dict['seq_truth'])):
                # TODO : une catégorie avec une majorité de gaps et sinon 
                if m_dict['mask_substitutions_ancestor'][k]==1:    # substitution
                    aas = ''
                    aas += human_str[k] + mouse_str[k] + anchor_str[k]
                    for a in m_dict['msa']:
                        aas += a[k]
                    x = Counter(aas)
                    nb_gaps = x['-']
                    if nb_gaps > int(len(aas)*0.95):    # special category if there are more than 95% gaps
                        subst_tot['gaps'] += 1
                        if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution)
                            subst_gud['gaps'] += 1
                    else:
                        x = Counter(aas.replace('-', '').replace('X', ''))
                        nb_diff = len(list(x.keys()))-1
                        subst_tot[str(nb_diff)] += 1
                        if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution)
                            subst_gud[str(nb_diff)] += 1
    subst = {k: subst_gud[k]/subst_tot[k] if subst_tot[k]>0 else 0 for k in keys}
    print('Number in each category:', [subst_tot[k] for k in keys])
    plt.bar(range(len(keys)), subst.values(), label="accuracy according to the conservation")
    plt.xticks(range(len(keys)), keys, rotation = 45)
    plt.legend()
    plt.show()
    return

    
def x6_accuracy_length(json_dicts):
    accuracies = [[] for i in range(80)]
    all_tot = [0 for i in range(80)]
    all_gud = [0 for i in range(80)]
    for i, m_dict in enumerate(json_dicts[-10000:]):
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys():
            tot = 0
            gud = 0
            for k in range(len(m_dict['seq_truth'])):
                if m_dict['mask_substitutions_ancestor'][k]==1:    # substitution
                    tot += 1
                    if int(m_dict['seq_pred'][k]) == int(m_dict['seq_truth'][k]):    # well-predicted substitution)
                        gud += 1
            if tot > 0:
                q = len(m_dict['seq_truth']) // 10
                accuracies[q].append(gud/tot)
                all_tot[q] += tot
                all_gud[q] += gud
    #plt.ylim(0.18,0.4)
    acc_means = [np.mean(np.array(accuracies[q])) for q in range(80)]
    plt.plot(np.array(range(80))*10, acc_means)
    plt.plot(np.array(range(80))*10, np.array(all_gud)/np.array(all_tot))
    plt.xlabel('seq length')
    plt.ylabel('accuracy')
    #plt.legend()
    plt.show()
    return


def x7_accuracy_matrix(json_dicts, normalize=None):
    matrix_pred = np.zeros((21,21)).astype('float64')
    matrix_tot  = np.ones((21,21)).astype('float64')*0.001   # to avoid 0 division
    for i, m_dict in enumerate(json_dicts[-2000:]):
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'seq_anchor' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys():
            for j in range(len(m_dict['seq_truth'])):
                if m_dict['mask_substitutions_ancestor'][j]:    # substitution
                    matrix_tot[int(m_dict['seq_truth'][j]), int(m_dict['seq_ancestor'][j])] += 1
                    if int(m_dict['seq_pred'][j]) == int(m_dict['seq_truth'][j]):    # well-predicted substitution)
                        matrix_pred[int(m_dict['seq_truth'][j]), int(m_dict['seq_ancestor'][j])] += 1
    matrix_subst = matrix_pred/matrix_tot
    with open('accuracy_matrix', 'wb') as f:
        np.save(f, matrix_subst)
    show_matrix_aa_to_image(matrix_subst, normalize=normalize)
    
def x7_accuracy_matrix_thresh(json_dicts, normalize=None):
    matrix_pred = np.zeros((21,21)).astype('float64')
    matrix_tot  = np.ones((21,21)).astype('float64')*0.001   # to avoid 0 division
    for i, m_dict in enumerate(json_dicts[-2000:]):
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'seq_anchor' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys():
            for j in range(len(m_dict['seq_truth'])):
                if m_dict['mask_substitutions_ancestor'][j]:    # substitution
                    matrix_tot[int(m_dict['seq_truth'][j]), int(m_dict['seq_ancestor'][j])] += 1
                    if int(m_dict['seq_pred'][j]) == int(m_dict['seq_truth'][j]):    # well-predicted substitution)
                        matrix_pred[int(m_dict['seq_truth'][j]), int(m_dict['seq_ancestor'][j])] += 1
    matrix_subst = matrix_pred/matrix_tot
    accuracies = []
    subst_percent = []
    mask_matrix = np.zeros((21,21)).astype('float64')
    matrix_max = np.copy(matrix_tot)
    for i in range(21):
        for j in range(21):
            if i != j:
                max_index = np.argmax(matrix_max)
                x_i, x_j = np.unravel_index(max_index, matrix_tot.shape)
                mask_matrix[x_i, x_j] = 1
                matrix_max[x_i, x_j] = 0
                accuracy = np.sum(matrix_pred*mask_matrix)/(np.sum(matrix_tot*mask_matrix))
                #print(np.sum(matrix_subst*mask_matrix))
                #print(np.sum(mask_matrix))
                percent_of_subst = np.sum(matrix_tot*mask_matrix)/np.sum(matrix_tot)
                #print(np.sum(matrix_tot*mask_matrix))
                #print(np.sum(matrix_tot))
                #print('Percent of substitutions:', percent_of_subst, 'Accuracy', accuracy)
                #input()
                accuracies.append(accuracy)
                subst_percent.append(percent_of_subst)

    print('accuracy:', accuracy)
    #for thresh in range(100):
    #    subst_tot_thresh = 0
    #    subst_gud_thresh = 0
    #    for i in range(21):
    #        for j in range(21):
    #            if matrix_subst[i,j] > thresh/100:
    #                subst_tot_thresh += matrix_tot[i,j]
    #                subst_gud_thresh += matrix_pred[i,j]
    #    percent_over_thresh = subst_tot_thresh / sum(sum(matrix_tot)) * 100
    #    print('There are', percent_over_thresh, '% of substitutions predicted with more than', subst_gud_thresh/subst_tot_thresh, '% accuracy (thresh=', thresh, ')')
    #    accuracies.append(subst_gud_thresh/subst_tot_thresh)
    #    subst_percent.append(percent_over_thresh)
    plt.plot(np.array(subst_percent), np.array(accuracies))
    #print(subst_percent)
    #print(accuracies)
    plt.xlabel('% of most common substitutions')
    plt.ylabel('accuracy')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
    return
    #with open('accuracy_matrix', 'wb') as f:
    #    np.save(f, matrix_subst)
    #show_matrix_aa_to_image(matrix_subst, normalize=normalize)


def x8_accuracy_matrix_properties(json_dicts, normalize=None):
    attribute_names = ['proline', 'aromatic', 'aliphatic', 'small', 'tiny', 'hydrophobic', 'polar', 'charged', 'negative', 'positive']
    m_len = len(attribute_names)
    matrix_pred = np.zeros((m_len, m_len)).astype('float64')
    matrix_tot  = np.ones((m_len, m_len)).astype('float64')*0.0001
    for i, m_dict in enumerate(json_dicts[-2000:]):
        if 'seq_truth' in m_dict.keys() and 'seq_pred' in m_dict.keys() and 'seq_anchor' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys():
            for j in range(len(m_dict['seq_truth'])):
                if m_dict['mask_substitutions_ancestor'][j]:    # substitution
                    aa_truth = residue_constants.aatype_to_str_sequence([int(m_dict['seq_truth'][j])])
                    aa_ancestor = residue_constants.aatype_to_str_sequence([int(m_dict['seq_ancestor'][j])])
                    attributes_truth = utils.aa_attributes(aa_truth)
                    attributes_ancestor = utils.aa_attributes(aa_ancestor)
                    for a_truth in attributes_truth:
                        for a_ancestor in attributes_ancestor:
                            matrix_tot[attribute_names.index(a_truth), attribute_names.index(a_ancestor)] += 1
                    if int(m_dict['seq_pred'][j]) == int(m_dict['seq_truth'][j]):    # well-predicted substitution)
                        for a_truth in attributes_truth:
                            for a_ancestor in attributes_ancestor:
                                matrix_pred[attribute_names.index(a_truth), attribute_names.index(a_ancestor)] += 1
    
    mat = matrix_pred/matrix_tot

    with open('accuracy_matrix.npy', 'wb') as f:
        np.save(f, mat)
        
    if normalize=='pred':
        row_sums = mat.sum(axis=0)
        mat = mat / row_sums[:, np.newaxis]
    if normalize=='truth':
        row_sums = mat.sum(axis=1)
        mat = mat / row_sums[:, np.newaxis]
    labels = attribute_names
    df_cm = pd.DataFrame(
        mat,
        index=labels,
        columns=labels,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.05, right=.65)
    sn.set(font_scale=1.5)
    sn.heatmap(df_cm, annot_kws={"size": 22}, fmt='d', ax=ax)
    plt.xlabel('Ancestor')
    plt.ylabel('Truth')
    #plt.savefig('matrix.svg')
    #with open('matrix.npy', 'wb') as f:
    #    np.save(f, mat)
    print(labels)
    plt.show()


    
json_dicts = open_jsons()

leave = False
keys = list(json_dicts[0].keys())
group_by_epoch = False
smooth=0
while not leave:
    print('\n\nPlease input what you wanna see:')
    for i, k in enumerate(keys):
        print(i, 'for', k)
    print('x1 for confusion matrix between truth and prediction on substitution positions')
    print('x1b for confusion matrix between truth and prediction on substitution positions (normalized by truth)')
    print('x1c for confusion matrix between truth and prediction on substitution positions (normalized by pred)')
    print('x2 accuracy substitutions (ancestor/target)')
    print('x3 accuracy substitutions (ancestor/target) for 50% most and least varrying sequences')
    print('x4 accuracy substitutions according to the position')
    print('x5 accuracy substitutions according to the conservation')
    print('x6 accuracy substitutions according to the length of the sequence')
    print('x7 accuracy substitutions for each aa pair')
    print('x7b accuracy substitutions for all aa pairs with accuracy over a threshold')
    print('x8 accuracy substitutions for each aa attribute')
    print('s to switch between raw data and epoch means')
    print('exit for exit')
    m_input = input()
    if m_input == 's':
        group_by_epoch = not group_by_epoch
        print("group_by_epoch:", group_by_epoch)
        continue
    if m_input == 'exit':
        leave = True
        continue
    try:
        if m_input not in ['x1', 'x1b', 'x1c', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x7b', 'x8']:
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
        print('accuracy (ancestor/beaver)')
        x2_accuracy_mask_substitutions_ancestor(json_dicts, group_by_epoch=group_by_epoch, smooth=0)
    elif m_input == 'x3':
        print('accuracy (ancestor/beaver) for most and least verrying sequences')
        x3_accuracy_most_least(json_dicts, smooth=50)
    elif m_input == 'x4':
        print('accuracy (ancestor/beaver) according to the position')
        x4_accuracy_position(json_dicts)
    elif m_input == 'x5':
        print('accuracy (ancestor/beaver) according to the conservation of aa')
        x5_accuracy_conservation(json_dicts)
    elif m_input == 'x6':
        print('accuracy (ancestor/beaver) according to the seq length')
        x6_accuracy_length(json_dicts)
    elif m_input == 'x7':
        print('accuracy (ancestor/beaver) for each pair of AA')
        x7_accuracy_matrix(json_dicts)
    elif m_input == 'x7b':
        print('accuracy (ancestor/beaver) for each pair of AA with a threshold on accuracy')
        x7_accuracy_matrix_thresh(json_dicts)
    elif m_input == 'x8':
        print('accuracy (ancestor/beaver) for attributes of AAs')
        x8_accuracy_matrix_properties(json_dicts)
    elif 'seq' in keys[int(m_input)]:
        plot_seq(keys[int(m_input)], json_dicts, group_by_epoch)
    else:
        print('Plotting the metric', keys[int(m_input)])
        plot_metric(keys[int(m_input)], json_dicts, group_by_epoch=group_by_epoch, smooth=smooth)
        
