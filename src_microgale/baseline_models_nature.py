import utils_baseline
import numpy as np
import random



class BaselineBlosum():
    def __init__(self, method='argmax'):
        # super().__init__()
        self.method = method
        if method == 'argmax':
            self.blosum = utils_baseline.blosum62_argmaxs()
        else:
            aa_list, subst_dict = utils_baseline.blosum62_freq()
            self.aa_list = aa_list
            self.subst_dict = subst_dict

    def predict_aa_substitution(self, aa):
        if aa == '-':
            return '-'
        if self.method == 'argmax':
            return self.blosum[aa]
        else:
            aa = np.random.choice([aa for aa in self.aa_list], p=self.subst_dict[aa])
            return str(aa)

    def forward(self, x):
        substituted_seq = [self.predict_aa_substitution(aa) for aa in x]
        return substituted_seq
    


class BaselineTransitionMatrix():
    def __init__(self, method='argmax'):
        self.method = method
        aas, subst_dict = utils_baseline.transitionMatrix()
        self.aa_list = aas
        self.subst_dict = subst_dict

    def predict_aa_substitution(self, aa):
        if aa == '-':
            return '-'
        if self.method == 'argmax':
            amax = np.argmax(self.subst_dict[aa])
            return self.aa_list[amax]
        else:
            aa = np.random.choice([aa for aa in self.aa_list], p=self.subst_dict[aa])
            return str(aa)

    def forward(self, x):
        substituted_seq = [self.predict_aa_substitution(aa) for aa in x]
        return substituted_seq



class BaselineGeneticCode():
    def __init__(self):
        self.genetic_code = utils_baseline.genetic_code()

    def predict_aa_substitution(self, aa):
        if aa == '-':
            return '-'
        aas = self.genetic_code[aa]
        aa = random.choice(aas)
        return str(aa)

    def forward(self, x):
        substituted_seq = [self.predict_aa_substitution(aa) for aa in x]
        return substituted_seq
