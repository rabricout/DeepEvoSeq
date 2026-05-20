import utils_baseline
import numpy as np

from collections import Counter
from tqdm import tqdm


class Baseline_position():
    def train(self, train_loader):
        print('> training model')
        tot = 0
        tot_true = 0
        for _, batch_p in tqdm(train_loader):
            batch_p = batch_p[0]
            tot += batch_p.shape[0]
            tot_true += batch_p.sum().item()
        self.subst_rate = tot_true/tot
    
    def set_species(self, ids_list, species):
        self.ids_dict = {x:i for i, x in enumerate(ids_list)}
        self.species = species


class BaselineRandom(Baseline_position):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        p = self.subst_rate
        subst_positions = np.random.choice(
            [0, 1],
            size=len(X[0][0]),
            p=[1 - p, p]
        )
        return subst_positions
    

class BaselineConsensus(Baseline_position):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X_a1 = X[self.ids_dict['A1']][0]
        X_others = [i[0] for i in X]
        del X_others[~self.ids_dict['A1']]
        consensus = ''
        for i in range(len(X_a1)):
            aas = [tmp_X[i] for tmp_X in X_others]
            consensus += Counter(aas).most_common(1)[0][0]
        subst = np.array([int(X_a1[i]!=consensus[i]) for i in range(len(X_a1))])
        return subst


class BaselineProxy(Baseline_position):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X_a1 = X[self.ids_dict['A1']][0]
        X_others = [i[0] for i in X]
        del X_others[~self.ids_dict['A1']]
        variability = []
        for i in range(len(X_a1)):
            aas = [tmp_X[i] for tmp_X in X_others]
            variability.append(len(set(aas)))
        max_var = max(variability)
        subst = np.array([int(variability[i] >= max_var-1) for i in range(len(variability))])
        return subst
    

class BaselineSpecies(Baseline_position):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X_a1 = X[self.ids_dict['A1']][0]
        X_species = X[self.ids_dict[self.species]][0]
        subst = np.array([X_a1[i]!=X_species[i] for i in range(len(X_a1))])
        return subst
    
