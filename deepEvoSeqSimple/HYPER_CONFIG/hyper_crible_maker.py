import numpy as np
import itertools
import random
import json

target_folder = 'CONFIG_FILES/'

nb_evoformer_layers = [1, 2, 4]
subst_rate_loss_pos_weight = [0.1, 0.5, 1]
activation = ['ReLU', 'Tanh', 'GELU']
norm = ['None', 'Layer']    # 'Batch' is probably useless because batch size is 1 distributed on different GPU
intern_dim_1 = [64, 128, 256]
intern_dim_2 = [32, 64, 128]
#additional_layers = [0, 1, 2]
#precision = ['bf16', '32']



elements_names = ['nb_evoformer_layers', 'subst_rate_loss_pos_weight', 'activation', 'norm', 'intern_dim_1', 'intern_dim_2']

all_configs = []
for i, x in enumerate(itertools.product(nb_evoformer_layers, subst_rate_loss_pos_weight, activation, norm, intern_dim_1, intern_dim_2)):
    m_dict = {}
    for j, e in enumerate(x):
        m_dict[elements_names[j]] = e
    json_object = json.dumps(m_dict, indent=4)
    all_configs.append(json_object)

nb_configs = 20
chosen_configs = random.sample(all_configs, nb_configs)
for i, config in enumerate(chosen_configs):
    with open(target_folder+str(i)+'.json', 'w') as outfile:
        outfile.write(config)
    
