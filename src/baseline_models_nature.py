import utils_baseline
import numpy as np
import random
import torch
import esm

from Bio import AlignIO



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
    

class BaselineLG():
    def __init__(self, method='argmax'):
        # super().__init__()
        self.method = method
        if method == 'argmax':
            self.lg = utils_baseline.lg_argmax()
        else:
            aa_list, subst_dict = utils_baseline.lg_freq()
            self.aa_list = aa_list
            self.subst_dict = subst_dict

    def predict_aa_substitution(self, aa):
        if aa == '-':
            return '-'
        if self.method == 'argmax':
            return self.lg[aa]
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


class BaselineESM2():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    def forward(self, x, p):
        # Load ESM2 model
        batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        sequence = list(x)
        for i, s in enumerate(p[0]):
            if s:
                sequence[i] = '<mask>'
        masked_sequence = "".join(sequence)
        data = [("protein1", masked_sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        # Inference
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)

        logits = results["logits"]  # shape: (batch, seq_len, vocab_size)
        pred_str = ''
        for i, l in enumerate(logits[0,1:-1]):
            if sequence[i] == '-':
                pred_str += '-'
            else:
                probs = torch.softmax(l, dim=-1)
                top1 = torch.topk(probs, 1)
                token = self.alphabet.get_tok(top1.indices.item())
                if token == 'X':
                    token = '-'
                pred_str += token
        return list(pred_str)


class BaselinePhyloBayesMPI():
    def __init__(self):
        pass

    def forward(self, x, p, ids):
        try:
            alignment = AlignIO.read('PB_MPI/RESULTS_eval/'+ids.split('_')[1]+'/chain_ppred9.ali', "phylip")
        except:
            return []
        for record in alignment:
            if 'SCIURUS' in record.id:
                pb_mpi_seq = str(record.seq)
        
        # l = list(pb_mpi_seq)
        # random.shuffle(l)
        # pb_mpi_seq = ''.join(l)
        pred_str = ''
        for i, _ in enumerate(p[0]):
            if p[0,i]:
                pred_str += pb_mpi_seq[i]
            else:
                pred_str += x[i]
        return list(pred_str)
