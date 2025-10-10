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
from sklearn import metrics
from sklearn.metrics import confusion_matrix

matplotlib.use('TkAgg')


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()


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
    

def x0_subst_rates(json_dicts, group_by_epoch=False):
    subst_diff_pred = []
    subst_pred = []
    subst_truth = []
    for m_dict in json_dicts:
        if 'substitution_rate_pred' in m_dict.keys() and 'substitution_rate_truth' in m_dict.keys():
            subst_diff_pred.append(np.abs(m_dict['substitution_rate_truth'] - m_dict['substitution_rate_pred']))
            subst_pred.append(m_dict['substitution_rate_pred'])
            subst_truth.append(m_dict['substitution_rate_truth'])
    print('Mean of thoretical subst rate truth', np.mean(np.array(subst_truth)))
    print('Mean of thoretical subst rate pred', np.mean(np.array(subst_pred)))
    plt.plot(subst_diff_pred, label='diff (abs) between pred and truth', linestyle=':', c='blue')
    plt.plot(subst_truth, label='truth subst rate (ancestor/beaver)', c='red')
    plt.plot(subst_pred, label='predicted subst rate (ancestor_beaver)', c='blue')
    plt.legend(loc='upper left')
    plt.title('Distance between predictions and truth of substitution rate')
    plt.show()

    
def x1_plot_f_score_binary(json_dicts, group_by_epoch=False, m_filter=None):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts[-10000:]):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            if 2*m_dict['tp']+m_dict['fp']+m_dict['fn'] > 0:
                epoch = m_dict['epoch_nb']
                f_score = 2*m_dict['tp'] / (2*m_dict['tp']+m_dict['fp']+m_dict['fn'])
                if group_by_epoch:
                    metric_data[epoch].append(f_score)
                else:
                    metric_data.append(f_score)
    
    print('Mean of f-score', np.mean(metric_data))
    print('Number of samples of f-score', len(metric_data))
    print('Std of f-score', np.std(metric_data))
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    plt.plot(metric_data)
    plt.title('F-score')
    plt.show()

    
def x1_plot_f_score(json_dicts, group_by_epoch=False, thresh=0.5, smooth=10, m_filter=None):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts[-10000:]):
        tp = 0
        tn = 0
        fn = 0
        fp = 0
        if 'mask_substitutions_ancestor' in m_dict.keys():    #'seq_pred_pos_sm' in m_dict.keys()
            seq = m_dict['mask_substitutions_ancestor']
            for j, v in enumerate(seq):
                if (m_filter=='mid' and j>50 and j<len(seq)-50) or (m_filter=='edges' and (j<50 or j>len(seq)-50)) or not m_filter:
                    #pred = int(m_dict['seq_pred_pos_sm'][j][1]>thresh)
                    pred = m_dict['seq_pred'][j]!=m_dict['seq_ancestor'][j]
                    if m_dict['mask_substitutions_ancestor'][j]==1 and pred==1:
                        tp += 1
                    if m_dict['mask_substitutions_ancestor'][j]==1 and pred==0:
                        fn += 1
                    if m_dict['mask_substitutions_ancestor'][j]==0 and pred==0:
                        tn += 1
                    if m_dict['mask_substitutions_ancestor'][j]==0 and pred==1:
                        fp += 1
        if 2*tp+fp+fn > 0:
            epoch = m_dict['epoch_nb']
            f_score = 2*tp / (2*tp+fp+fn)
            if group_by_epoch:
                metric_data[epoch].append(f_score)
            else:
                metric_data.append(f_score)
    print('Mean of f-score (', m_filter, ')', np.mean(metric_data))
    print('Number of samples of f-score (', m_filter, ')', len(metric_data))
    print('Std of f-score (', m_filter, ')', np.std(metric_data))
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    plt.plot(metric_data)
    plt.title('F-score')
    plt.show()

    
def x2_plot_precision_recall(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data_x = [[] for _ in range(epoch_max)] if group_by_epoch else []
    metric_data_y = [[] for _ in range(epoch_max)] if group_by_epoch else []
    colors = [[] for _ in range(epoch_max)] if group_by_epoch else []
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i, m_dict in enumerate(json_dicts[-2000:]):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            tp += m_dict['tp']
            tn += m_dict['tn']
            fp += m_dict['fp']
            fn += m_dict['fn']
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
    print('precision:', tp/(tp+fp))
    print('recall:', tp/(tp+fn))
    print('plotted values', metric_data_x, metric_data_y)
    plt.scatter(metric_data_x, metric_data_y, s=10+200/np.sqrt(len(metric_data_x)), c=colors, cmap='viridis')
    plt.colorbar(label='epochs')
    plt.title('Precision-recall')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.show()

def x2b_plot_precision_recall_curve(json_dicts, group_by_epoch=False):
    precision = []
    recall = []
    for thresh in np.arange(0, 1, 0.05):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i, m_dict in enumerate(json_dicts[-1000:]):
            if 'seq_pred_pos_sm' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys():
                for j, v in enumerate(m_dict['mask_substitutions_ancestor']):
                    pred = int(m_dict['seq_pred_pos_sm'][j][1]>thresh)
                    if m_dict['mask_substitutions_ancestor'][j]==1 and pred==1:
                        tp += 1
                    if m_dict['mask_substitutions_ancestor'][j]==1 and pred==0:
                        fn += 1
                    if m_dict['mask_substitutions_ancestor'][j]==0 and pred==0:
                        tn += 1
                    if m_dict['mask_substitutions_ancestor'][j]==0 and pred==1:
                        fp += 1
        if (tp+fp) > 0 and (tp+fn) > 0:
            precision.append(tp / (tp+fp))
            recall.append(tp / (tp+fn))
        if thresh == 0.5:
            if (tp+fp) > 0:
                print('precision:', tp/(tp+fp))
            if tp/(tp+fn) > 0:
                print('recall:', tp/(tp+fn))
        #else:
        #    precision.append(1)
        #    recall.append(0)
    #plt.step(precision, recall)
    print('AUC', metrics.auc(recall, precision))
    plt.plot(recall, precision)
    plt.scatter([0.058], [0.055], color='orange')    # random
    #plt.scatter([0.413], [0.132], color='red')    # mouse
    plt.scatter([0.462], [0.157], color='red')    # mouse
    #plt.scatter([0.108], [0.130], color='purple')    # proxy ancestor
    plt.scatter([0.163], [0.188], color='purple')    # proxy ancestor
    plt.legend(['neural network', 'mouse', 'random', 'proxy'])
    plt.title('Precision/recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.show()
    

def x3_plot_ROC(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data_x = [[] for _ in range(epoch_max)] if group_by_epoch else []
    metric_data_y = [[] for _ in range(epoch_max)] if group_by_epoch else []
    colors = [[] for _ in range(epoch_max)] if group_by_epoch else []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, m_dict in enumerate(json_dicts[-2000:]):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            tp += m_dict['tp']
            tn += m_dict['tn']
            fp += m_dict['fp']
            fn += m_dict['fn']
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
    
    print('fpr:', fp/(fp+tn))
    print('tpr:', tp/(tp+fn))
    print('plotted values', metric_data_x, metric_data_y)
    plt.scatter(metric_data_x, metric_data_y, s=10+200/np.sqrt(len(metric_data_x)), c=colors, cmap='viridis')
    plt.colorbar(label='training steps')
    plt.plot([0, 1], [0, 1], linestyle='-')
    plt.plot([0, 0.045], [1, 0], linestyle='-', color='red')
    plt.title('ROC')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.show()


def x3b_plot_ROC_curve(json_dicts, group_by_epoch=False):
    fpr = []
    tpr = []
    for thresh in np.arange(0, 1, 0.05):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i, m_dict in enumerate(json_dicts[-2000:]):
            if 'seq_pred_pos_sm' in m_dict.keys() and 'mask_substitutions_ancestor' in m_dict.keys():
                for j, v in enumerate(m_dict['mask_substitutions_ancestor']):
                    pred = int(m_dict['seq_pred_pos_sm'][j][1]>thresh)
                    if m_dict['mask_substitutions_ancestor'][j]==1 and pred==1:
                        tp += 1
                    if m_dict['mask_substitutions_ancestor'][j]==1 and pred==0:
                        fn += 1
                    if m_dict['mask_substitutions_ancestor'][j]==0 and pred==0:
                        tn += 1
                    if m_dict['mask_substitutions_ancestor'][j]==0 and pred==1:
                        fp += 1
        if (fp+tn) > 0 and (tp+fn) > 0:
            fpr.append(fp / (fp+tn))
            tpr.append(tp / (tp+fn))
        else:
            fpr.append(0)
            tpr.append(0)
    #plt.step(fpr, tpr)
    plt.plot(fpr, tpr)
    
    print('AUC', metrics.auc(fpr, tpr))
    #plt.scatter([0.116], [0.437], color='red')    # our
    #plt.scatter([0.148], [0.411], color='red')    # mouse
    plt.scatter([0.105], [0.443], color='red')    # mouse
    #plt.scatter([0.106], [0.312], color='green')    # human
    plt.scatter([0.056], [0.055], color='orange')    # random
    #plt.scatter([0.075], [0.078], color='orange')    # random
    plt.scatter([0.0341], [0.163], color='purple')    # proxy ancestor
    #plt.scatter([0.0319], [0.0999], color='purple')    # proxy ancestor
    #plt.legend(['neural network', 'mouse', 'human', 'random', 'proxy'])
    plt.legend(['neural network', 'mouse', 'random', 'proxy'])
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    plt.title('ROC curve')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.show()
    

def x4_plot_subst_positions_binary(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    subst_pos_truth = [np.zeros((100)) for _ in range(epoch_max)] if group_by_epoch else np.zeros((100))
    subst_pos_pred = [np.zeros((100)) for _ in range(epoch_max)] if group_by_epoch else np.zeros((100))
    tot_pos = [np.zeros((100)) for _ in range(epoch_max)] if group_by_epoch else np.zeros((100))
    for m_dict in json_dicts:
        if 'seq_ancestor' in m_dict.keys():
            length = len(m_dict['seq_ancestor'])
            epoch = m_dict['epoch_nb']
            for i in range(length):
                local_bin = int(i/length*100)
                if group_by_epoch:
                    subst_pos_truth[epoch][local_bin] += m_dict['mask_substitutions_ancestor'][i]
                    subst_pos_pred[epoch][local_bin]  += m_dict['seq_pred_pos'][i]
                    tot_pos[epoch][local_bin] += 1
                elif epoch_max==1 or m_dict['epoch_nb'] > 0:
                    subst_pos_truth[local_bin] += m_dict['mask_substitutions_ancestor'][i]
                    subst_pos_pred[local_bin]  += m_dict['seq_pred_pos'][i]
                    tot_pos[local_bin] += 1
    if group_by_epoch:
        metric_data_truth = [subst_pos_truth[i] / tot_pos[i] for i in range(epoch_max)]
        metric_data_pred  = [subst_pos_pred[i]  / tot_pos[i] for i in range(epoch_max)]
        for i in range(epoch_max):
            plt.plot(metric_data_truth[i], label='truth '+str(i))
            plt.plot(metric_data_pred[i], label='predictions '+str(i), linestyle='dashed')
        print('Mean of subst rate (truth)', np.mean(np.mean(metric_data_truth)))
        print('Mean of subst rate (pred)', np.mean(np.mean(metric_data_pred)))
    else:
        metric_data_truth = subst_pos_truth / tot_pos
        metric_data_pred = subst_pos_pred / tot_pos
        plt.plot(metric_data_truth, label='truth')
        plt.plot(metric_data_pred, label='predictions', linestyle='dashed')
        print('Mean of subst rate (truth)', np.mean(metric_data_truth))
        print('Mean of subst rate (pred)', np.mean(metric_data_pred))
    plt.legend(loc='upper left')
    plt.title('Substitutions according to position')
    plt.show()
    

def x4_plot_subst_positions(json_dicts, group_by_epoch=False, thresh=0.5):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    subst_pos_truth = [np.zeros((100)) for _ in range(epoch_max)] if group_by_epoch else np.zeros((100))
    subst_pos_pred = [np.zeros((100)) for _ in range(epoch_max)] if group_by_epoch else np.zeros((100))
    tot_pos = [np.zeros((100)) for _ in range(epoch_max)] if group_by_epoch else np.zeros((100))
    for m_dict in json_dicts:
        if 'seq_ancestor' in m_dict.keys(): # and 'seq_pred_pos_sm' in m_dict.keys():
            length = len(m_dict['seq_ancestor'])
            epoch = m_dict['epoch_nb']
            for i in range(length):
                local_bin = int(i/length*100)
                if 'seq_pred_pos_sm' in m_dict.keys():
                    pred = int(m_dict['seq_pred_pos_sm'][i][1]>thresh)
                else:
                    pred = m_dict['seq_pred'][i]!=m_dict['seq_ancestor'][i]
                if group_by_epoch:
                    subst_pos_truth[epoch][local_bin] += m_dict['mask_substitutions_ancestor'][i]
                    subst_pos_pred[epoch][local_bin]  += pred
                    tot_pos[epoch][local_bin] += 1
                elif epoch_max==1 or m_dict['epoch_nb'] > 0:
                    subst_pos_truth[local_bin] += m_dict['mask_substitutions_ancestor'][i]
                    subst_pos_pred[local_bin]  += pred
                    tot_pos[local_bin] += 1
    if group_by_epoch:
        metric_data_truth = [subst_pos_truth[i] / tot_pos[i] for i in range(epoch_max)]
        metric_data_pred  = [subst_pos_pred[i]  / tot_pos[i] for i in range(epoch_max)]
        for i in range(epoch_max):
            plt.plot(metric_data_truth[i], label='truth '+str(i))
            plt.plot(metric_data_pred[i], label='predictions '+str(i)+' with thresh='+str(thresh), linestyle='dashed')
        print('Mean of subst rate (truth)', np.mean(np.mean(metric_data_truth)))
        print('Mean of subst rate (pred)', np.mean(np.mean(metric_data_pred)))
    else:
        metric_data_truth = subst_pos_truth / tot_pos
        metric_data_pred = subst_pos_pred / tot_pos
        plt.plot(metric_data_truth, label='truth')
        plt.plot(metric_data_pred, label='predictions', linestyle='dashed')
        print('Mean of subst rate (truth)', np.mean(metric_data_truth))
        print('Mean of subst rate (pred)', np.mean(metric_data_pred))
    plt.legend(loc='upper left')
    plt.ylim(0,0.25)
    plt.title('Substitutions according to position')
    plt.show()


def x5_phi_binary(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts[-3000:]):
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
    
    print('Number of samples of phi', len(metric_data))
    print('Std of phi', np.std(metric_data))
    print('Mean of phi', np.mean(metric_data))
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    #print('Mean of phi', np.mean(metric_data))
    plt.plot(metric_data)
    plt.title('phi')
    plt.show()

    
def x5_phi(json_dicts, group_by_epoch=False, thresh=0.5, m_filter=None):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts[-2000:]):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        if 'mask_substitutions_ancestor' in m_dict.keys():    #and 'seq_pred_pos_sm' in m_dict.keys()
            seq = m_dict['mask_substitutions_ancestor']
            for j, v in enumerate(seq):
                if (m_filter=='mid' and j>50 and j<len(seq)-50) or (m_filter=='edges' and (j<50 or j>len(seq)-50)) or not m_filter:
                    #pred = int(m_dict['seq_pred_pos_sm'][j][1]>thresh)
                    pred = m_dict['seq_pred'][j]!=m_dict['seq_ancestor'][j]
                    if m_dict['mask_substitutions_ancestor'][j]==1 and pred==1:
                        tp += 1
                    if m_dict['mask_substitutions_ancestor'][j]==1 and pred==0:
                        fn += 1
                    if m_dict['mask_substitutions_ancestor'][j]==0 and pred==0:
                        tn += 1
                    if m_dict['mask_substitutions_ancestor'][j]==0 and pred==1:
                        fp += 1
        N = tn + tp + fn + fp
        if N>0:
            S = (tp+fn)/N
            P = (tp+fp)/N
            if np.sqrt(P*S*(1-S)*(1-P)) > 0:
                epoch = m_dict['epoch_nb']
                phi = (tp/N-S*P) / np.sqrt(P*S*(1-S)*(1-P))
                if group_by_epoch:
                    metric_data[epoch].append(phi)
                else:
                    metric_data.append(phi)
    
    print('Number of samples of phi', len(metric_data))
    print('Std of phi', np.std(metric_data))
    print('Mean of phi', np.mean(metric_data))
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    plt.plot(metric_data)
    plt.title('phi')
    plt.show()

    
def x6_youdenJ(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts[-2000:]):
        if 'tp' in m_dict.keys() and 'tn' in m_dict.keys() and 'fp' in m_dict.keys() and 'fn' in m_dict.keys():
            if m_dict['tp']+m_dict['fn'] > 0 and m_dict['tn']+m_dict['fp'] > 0:
                epoch = m_dict['epoch_nb']
                j = m_dict['tp']/(m_dict['tp']+m_dict['fn']) + m_dict['tn']/(m_dict['tn']+m_dict['fp']) - 1
                if group_by_epoch:
                    metric_data[epoch].append(j)
                else:
                    metric_data.append(j)

    print('Number of samples of Youdens J', len(metric_data))
    print('Mean of Youdens J', np.mean(metric_data))
    print('Std of Youdens J', np.std(metric_data))
    if group_by_epoch:
        metric_data = [np.mean(d) for d in metric_data]
    else:
        metric_data = np.convolve(metric_data, np.ones(smooth)/smooth, mode='valid')
    plt.plot(metric_data)
    plt.title('Youden J')
    plt.show()

    
def x7_cohenKappa(json_dicts, group_by_epoch=False):
    epoch_max = int(json_dicts[-1]['epoch_nb'])+1
    metric_data = [[] for _ in range(epoch_max)] if group_by_epoch else []
    for i, m_dict in enumerate(json_dicts[-3000:]):
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
    print('Mean of Cohen Kappa', np.mean(metric_data))
    plt.plot(metric_data)
    plt.title('Cohen kappa')
    plt.show()


def x8_subst_rate_by_sequence(json_dicts):
    true_subst_rate = []
    pred_subst_rate_0_5 = []
    pred_subst_rate_0_75 = []
    pred_subst_rate = []
    for i, m_dict in enumerate(json_dicts[-1000:]):
        true_subst = 0
        pred_subst = 0
        pred_subst_0_5 = 0
        pred_subst_0_75 = 0
        if True: #'seq_pred_pos_sm' in m_dict.keys():
            seq = m_dict['mask_substitutions_ancestor']
            for j, v in enumerate(seq):
                if 'seq_pred_pos_sm' in m_dict.keys():
                    pred = int(m_dict['seq_pred_pos_sm'][j][1]>0.5)
                    pred_subst_0_5 += pred
                    pred = int(m_dict['seq_pred_pos_sm'][j][1]>0.75)
                    pred_subst_0_75 += pred
                else:
                    pred = m_dict['seq_pred'][j]!=m_dict['seq_ancestor'][j]
                    pred_subst += pred
                
                true_subst += m_dict['mask_substitutions_ancestor'][j]==1
        if 'seq_pred_pos_sm' in m_dict.keys():
            pred_subst_rate_0_5.append(pred_subst_0_5/len(seq))
            pred_subst_rate_0_75.append(pred_subst_0_75/len(seq))
        else:
            pred_subst_rate.append(pred_subst/len(seq))
        true_subst_rate.append(true_subst/len(seq))
    if 'seq_pred_pos_sm' in m_dict.keys():
        pearson_correlation = scipy.stats.pearsonr(true_subst_rate, pred_subst_rate_0_5)
        print('pearson_correlation (thresh=0.5)', pearson_correlation)
        pearson_correlation = scipy.stats.pearsonr(true_subst_rate, pred_subst_rate_0_75)
        print('pearson_correlation (thresh=0.75)', pearson_correlation)
        plt.scatter(true_subst_rate, pred_subst_rate_0_5, color='green',  s=5, label='threshold=0.5')
        plt.scatter(true_subst_rate, pred_subst_rate_0_75, color='orange', s=5, label='threshold=0.75')
    else:
        pearson_correlation = scipy.stats.pearsonr(true_subst_rate, pred_subst_rate)
        print('pearson_correlation', pearson_correlation)
        plt.scatter(true_subst_rate, pred_subst_rate, color='green',  s=5, label='mouse')
    plt.xlim(0,0.35)
    plt.ylim(0,0.6)
    plt.legend(loc='upper right')
    plt.xlabel('true substitution rate')
    plt.ylabel('predicted substitution rate')
    plt.title('True vs Predicted substitution rate based on the positions of substitutions')
    plt.show()



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
    print('x0 substitution rates')
    print('x1 f-score')
    print('x2 precision-recall')
    print('x2b precision-recall curve for last 1000 iter')
    print('x3 ROC values')
    print('x3b to plot ROC curve for last 1000 iter')
    print('x4 substitution positions')
    print('x5 phi coef')
    print('x6 Youdens J coef')
    print('x7 Cohen Kappa coef')
    print('x8 comparaison of substitution rate based on position predictions')
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
        if m_input not in ['x0', 'x1', 'x2', 'x2b', 'x3', 'x3b', 'x4', 'x5', 'x5', 'x6', 'x7', 'x8']:
            x = int(m_input)
            if x>len(keys)-1:
                print('Please input a proper number')
                continue
    except:
        print('Please enter a number')
        continue

    if m_input == 'x0':
        print('substitution rates')
        x0_subst_rates(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x1':
        print('plot_F_score')
        x1_plot_f_score_binary(json_dicts, group_by_epoch=group_by_epoch)
        #x1_plot_f_score(json_dicts, group_by_epoch=group_by_epoch, m_filter=None)
        #x1_plot_f_score(json_dicts, group_by_epoch=group_by_epoch, m_filter='mid')
        #x1_plot_f_score(json_dicts, group_by_epoch=group_by_epoch, m_filter='edges')
    elif m_input == 'x2':
        print('plot_precision_recall_values')
        x2_plot_precision_recall(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x2b':
        print('plot_precision_recall_curve')
        x2b_plot_precision_recall_curve(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x3':
        print('plot_ROC_value')
        x3_plot_ROC(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x3b':
        print('plot_ROC_curve')
        x3b_plot_ROC_curve(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x4':
        print('plot_subst_positions')
        x4_plot_subst_positions(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x5':
        print('(positions) phi coef')
        x5_phi_binary(json_dicts, group_by_epoch=group_by_epoch)
        #x5_phi(json_dicts, group_by_epoch=group_by_epoch, m_filter='mid')
        #x5_phi(json_dicts, group_by_epoch=group_by_epoch, m_filter='edges')
    elif m_input == 'x6':
        print('(positions) Youden J statitic')
        x6_youdenJ(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x7':
        print('(positions) Cohen Kappa')
        x7_cohenKappa(json_dicts, group_by_epoch=group_by_epoch)
    elif m_input == 'x8':
        print('(positions) substitution rates')
        x8_subst_rate_by_sequence(json_dicts)
    else:
        print('Plotting the metric', keys[int(m_input)])
        plot_metric(keys[int(m_input)], json_dicts, group_by_epoch=group_by_epoch, smooth=smooth)
        
