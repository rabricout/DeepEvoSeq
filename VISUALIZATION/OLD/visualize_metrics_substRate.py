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



    
def x1_plot_subst_rate_diff(json_dicts):
    subst_diff_pred = []
    subst_diff_proxy = []
    subst_pred = []
    subst_truth = []
    subst_human = []
    subst_proxy = []
    for m_dict in json_dicts:
        if 'substitution_rate_pred' in m_dict.keys() and 'substitution_rate_truth' in m_dict.keys() and 'substitution_rate_HA' in m_dict.keys():
            subst_diff_pred.append(np.abs(m_dict['substitution_rate_truth'] - m_dict['substitution_rate_pred']))
            subst_diff_proxy.append(np.abs(m_dict['substitution_rate_truth'] - m_dict['substitution_rate_HA']/0.045*0.027))
            if m_dict['substitution_rate_truth'] > 0.5:
                print('seqs')
                s_t = residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_truth']).astype(int))
                s_a = residue_constants.aatype_to_str_sequence(np.array(m_dict['seq_anchor']).astype(int))
                print("m_dict['seq_truth']", s_t)
                print("m_dict['seq_anchor']", s_a)
            subst_pred.append(m_dict['substitution_rate_pred'])
            subst_truth.append(m_dict['substitution_rate_truth'])
            subst_human.append(m_dict['substitution_rate_HA'])
            subst_proxy.append(m_dict['substitution_rate_HA']/0.045*0.027)
    plt.plot(subst_diff_pred, label='diff (abs) between pred and truth', linestyle=':', c='blue')
    plt.plot(subst_diff_proxy, label='diff (abs) between proxy and truth', linestyle=':', c='green')
    plt.plot(subst_truth, label='truth subst rate (ancestor/beaver)', c='red')
    plt.plot(subst_pred, label='predicted subst rate (ancestor_beaver)', c='blue')
    plt.plot(subst_proxy, label='proxy of subst rate CA (ancestor_beaver)', c='green')
    plt.plot(subst_human, label='truth subst rate HA (human/ancestor)', linestyle='--', c='black')
    plt.legend(loc='upper left')
    plt.title('Distance between predictions and truth of substitution rate')
    plt.show()

        
def x2_print_correlations(json_dicts):
    subst_truth = []
    subst_pred = []
    subst_HA = []
    last_1000_dicts = json_dicts[-1000:] if len(json_dicts) > 1000 else json_dicts
    for m_dict in last_1000_dicts:
        if 'substitution_rate_truth' in m_dict.keys() and 'substitution_rate_pred' in m_dict.keys() and 'substitution_rate_HA' in m_dict.keys():
            subst_truth.append(m_dict['substitution_rate_truth'])
            subst_pred.append(m_dict['substitution_rate_pred'])
            subst_HA.append(m_dict['substitution_rate_HA'])
    subst_truth = np.array(subst_truth)
    subst_pred = np.array(subst_pred)
    subst_HA = np.array(subst_HA)
    subst_proxy = subst_HA /0.045*0.027 #mean_HA * mean_CA
    pearson_correlation_truth_pred = scipy.stats.pearsonr(subst_truth, subst_pred)
    pearson_correlation_truth_proxy = scipy.stats.pearsonr(subst_truth, subst_proxy)
    
    diff_pred_truth_L1  = np.linalg.norm((subst_truth-subst_pred), ord=1) / len(subst_truth-subst_pred)
    diff_pred_truth_L2  = np.linalg.norm((subst_truth-subst_pred), ord=2) / len(subst_truth-subst_pred)
    diff_proxy_truth_L1 = np.linalg.norm((subst_truth-subst_proxy), ord=1) / len(subst_truth-subst_proxy)
    diff_proxy_truth_L2 = np.linalg.norm((subst_truth-subst_proxy), ord=2) / len(subst_truth-subst_proxy)

    print('pearson_correlation between predicted and truth', pearson_correlation_truth_pred)
    print('pearson_correlation between proxy and truth', pearson_correlation_truth_proxy, '\n')
    print('mean distance (L1 norm) between pred and truth', diff_pred_truth_L1)
    print('mean distance (L1 norm) between proxy and truth', diff_proxy_truth_L1, '\n')
    print('mean distance (L2 norm) between pred and truth', diff_pred_truth_L2)
    print('mean distance (L2 norm) between proxy and truth', diff_proxy_truth_L2)
    input()

    
def x3_plot_subst_rates(json_dicts):
    subst_rate_truth = []
    subst_rate_pred = []
    subst_rate_HA = []
    subst_rate_HT = []
    subst_rate_AT = []
    for m_dict in json_dicts:
        if 'substitution_rate_truth' in m_dict.keys() and 'substitution_rate_pred' in m_dict.keys() and 'substitution_rate_HA' in m_dict.keys():
            subst_rate_truth.append(m_dict['substitution_rate_truth'])
            subst_rate_pred.append(m_dict['substitution_rate_pred'])
            subst_rate_HA.append(m_dict['substitution_rate_HA'])
            subst_rate_HT.append(m_dict['substitution_rate_HT'])
            subst_rate_AT.append(m_dict['substitution_rate_AT'])
    print('Mean of HA (human/ancestor)', np.mean(np.array(subst_rate_HA)))
    print('Mean of HT (human/target)', np.mean(np.array(subst_rate_HT)))
    print('Mean of AT (ancestor/target)', np.mean(np.array(subst_rate_AT)))
    plt.plot(subst_rate_truth, label='subst_rate_truth')
    plt.plot(subst_rate_pred, label='subst_rate_pred')
    plt.plot(subst_rate_HA, label='subst_rate_HA')
    plt.plot(subst_rate_HT, label='subst_rate_HT')
    plt.plot(subst_rate_AT, label='subst_rate_AT')
    plt.legend(loc='upper left')
    plt.title('substitution_rates')
    plt.show()


def x4_subst_rates_scatter(json_dicts):
    subst_rate_truth = []
    subst_rate_pred = []
    subst_rate_HA = []
    for m_dict in json_dicts:
        if 'substitution_rate_truth' in m_dict.keys() and 'substitution_rate_pred' in m_dict.keys() and 'substitution_rate_HA' in m_dict.keys():
            subst_rate_truth.append(m_dict['substitution_rate_truth'])
            subst_rate_pred.append(m_dict['substitution_rate_pred'])
            subst_rate_HA.append(m_dict['substitution_rate_HA'])
    subst_rate_proxy = np.array(subst_rate_HA) / 0.045*0.027 #mean_HA * mean_CA
    plt.scatter(subst_rate_truth, subst_rate_pred, label='prediction with neural network', s=5)
    plt.scatter(subst_rate_truth, subst_rate_proxy, label='prediction with proxy', s=5)
    lim = 0.3
    plt.plot([0,lim],[0,lim], color='black')
    plt.legend(loc='upper left')
    plt.xlabel('true substitution rate')
    plt.ylabel('predicted substitution rate')
    plt.title('scatter plot comparing neural network predictions with proxy')
    plt.ylim(0,lim)
    plt.xlim(0,lim)
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
    print('x1 (substRate) for difference between substitution rate truth and predicted')
    print('x2 (substRate) for correlation between subst_rate CA (castor/ancestor) and subst_rate HA (humain/marmotte)')
    print('x3 (substRate) for subst_rates')
    print('x4 (substRate) for subst rates in cloud (pred vs truth)')
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
        if m_input not in ['x1', 'x2', 'x3', 'x4']:
            x = int(m_input)
            if x>len(keys)-1:
                print('Please input a proper number')
                continue
    except:
        print('Please enter a number')
        continue

    if m_input == 'x1':
        print('plot_difference of substitution rate between truth and prediction')
        x1_plot_subst_rate_diff(json_dicts)
    elif m_input == 'x2':
        print('correlations of substitution rates (proxy for branch length for the 1000 last data)')
        x2_print_correlations(json_dicts)
    elif m_input == 'x3':
        print('substitution rates (proxy for branch length)')
        x3_plot_subst_rates(json_dicts)
    elif m_input == 'x4':
        print('subst rates cloud (truth vs pred)')
        x4_subst_rates_scatter(json_dicts)
    else:
        print('Plotting the metric', keys[int(m_input)])
        plot_metric(keys[int(m_input)], json_dicts, group_by_epoch=group_by_epoch, smooth=smooth)
        
