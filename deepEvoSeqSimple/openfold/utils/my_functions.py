from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio import SeqIO
import subprocess
from io import StringIO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def my_align_seqs(seqs):
    seq_str = ''
    for i, seq in enumerate(seqs):
        seq_str += '>' + seq.description + '\n'
        seq_str += str(seq.seq) + '\n'
    child = subprocess.Popen(['mafft', '--anysymbol', '--quiet', '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    child.stdin.write(seq_str.encode())
    child_out = child.communicate()[0].decode('utf8')
    seq_ali = list(SeqIO.parse(StringIO(child_out), 'fasta'))
    child.stdin.close()
    return seq_ali


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,int)):
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



def f1(y_true, y_pred):
    y_pred = torch.round(y_pred)
    tp = torch.sum((y_true * y_pred).float(), dim=0)
    tn = torch.sum(((1 - y_true) * (1 - y_pred)).float(), dim=0)
    fp = torch.sum(((1 - y_true) * y_pred).float(), dim=0)
    fn = torch.sum((y_true * (1 - y_pred)).float(), dim=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)

    f1 = 2 * p * r / (p + r + 1e-7)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return torch.mean(f1)

def f1_loss(y_true, y_pred):
    tp = torch.sum((y_true * y_pred).float(), dim=0)
    tn = torch.sum(((1 - y_true) * (1 - y_pred)).float(), dim=0)
    fp = torch.sum(((1 - y_true) * y_pred).float(), dim=0)
    fn = torch.sum((y_true * (1 - y_pred)).float(), dim=0)
    
    #p = tp / (tp + fp + 1e-7)
    #r = tp / (tp + fn + 1e-7)

    #f1 = 2 * p * r / (p + r + 1e-7)
    #f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    f1 = 2 * tp / (2*tp+fp+fn+1e-7)
    return 1 - torch.mean(f1)




def blosum62_freq():
    aas = 'ARNDCQEGHILKMFPSTWYV'
    aas_dict = {aas[i]: i for i in range(len(aas))}
    matrix = [[0.0215, 0.0023, 0.0019, 0.0022, 0.0016, 0.0019, 0.0030, 0.0058, 0.0011, 0.0032, 0.0044, 0.0033, 0.0013, 0.0016, 0.0022, 0.0063, 0.0037, 0.0004, 0.0013, 0.0051],
              [0.0023, 0.0178, 0.0020, 0.0016, 0.0004, 0.0025, 0.0027, 0.0017, 0.0012, 0.0012, 0.0024, 0.0062, 0.0008, 0.0009, 0.0010, 0.0023, 0.0018, 0.0003, 0.0009, 0.0016],
              [0.0019, 0.0020, 0.0141, 0.0037, 0.0004, 0.0015, 0.0022, 0.0029, 0.0014, 0.0010, 0.0014, 0.0024, 0.0005, 0.0008, 0.0009, 0.0031, 0.0022, 0.0002, 0.0007, 0.0012],
              [0.0022, 0.0016, 0.0037, 0.0213, 0.0004, 0.0016, 0.0049, 0.0025, 0.0010, 0.0012, 0.0015, 0.0024, 0.0005, 0.0008, 0.0012, 0.0028, 0.0019, 0.0002, 0.0006, 0.0013],
              [0.0016, 0.0004, 0.0004, 0.0004, 0.0119, 0.0003, 0.0004, 0.0008, 0.0002, 0.0011, 0.0016, 0.0005, 0.0004, 0.0005, 0.0004, 0.0010, 0.0009, 0.0001, 0.0003, 0.0014],
              [0.0019, 0.0025, 0.0015, 0.0016, 0.0003, 0.0073, 0.0035, 0.0014, 0.0010, 0.0009, 0.0016, 0.0031, 0.0007, 0.0005, 0.0008, 0.0019, 0.0014, 0.0002, 0.0007, 0.0012],
              [0.0030, 0.0027, 0.0022, 0.0049, 0.0004, 0.0035, 0.0161, 0.0019, 0.0014, 0.0012, 0.0020, 0.0041, 0.0007, 0.0009, 0.0014, 0.0030, 0.0020, 0.0003, 0.0009, 0.0017],
              [0.0058, 0.0017, 0.0029, 0.0025, 0.0008, 0.0014, 0.0019, 0.0378, 0.0010, 0.0014, 0.0021, 0.0025, 0.0007, 0.0012, 0.0014, 0.0038, 0.0022, 0.0004, 0.0008, 0.0018],
              [0.0011, 0.0012, 0.0014, 0.0010, 0.0002, 0.0010, 0.0014, 0.0010, 0.0093, 0.0006, 0.0010, 0.0012, 0.0004, 0.0008, 0.0005, 0.0011, 0.0007, 0.0002, 0.0015, 0.0006],
              [0.0032, 0.0012, 0.0010, 0.0012, 0.0011, 0.0009, 0.0012, 0.0014, 0.0006, 0.0184, 0.0114, 0.0016, 0.0025, 0.0030, 0.0010, 0.0017, 0.0027, 0.0004, 0.0014, 0.0120],
              [0.0044, 0.0024, 0.0014, 0.0015, 0.0016, 0.0016, 0.0020, 0.0021, 0.0010, 0.0114, 0.0371, 0.0025, 0.0049, 0.0054, 0.0014, 0.0024, 0.0033, 0.0007, 0.0022, 0.0095],
              [0.0033, 0.0062, 0.0024, 0.0024, 0.0005, 0.0031, 0.0041, 0.0025, 0.0012, 0.0016, 0.0025, 0.0161, 0.0009, 0.0009, 0.0016, 0.0031, 0.0023, 0.0003, 0.0010, 0.0019],
              [0.0013, 0.0008, 0.0005, 0.0005, 0.0004, 0.0007, 0.0007, 0.0007, 0.0004, 0.0025, 0.0049, 0.0009, 0.0040, 0.0012, 0.0004, 0.0009, 0.0010, 0.0002, 0.0006, 0.0023],
              [0.0016, 0.0009, 0.0008, 0.0008, 0.0005, 0.0005, 0.0009, 0.0012, 0.0008, 0.0030, 0.0054, 0.0009, 0.0012, 0.0183, 0.0005, 0.0012, 0.0012, 0.0008, 0.0042, 0.0026],
              [0.0022, 0.0010, 0.0009, 0.0012, 0.0004, 0.0008, 0.0014, 0.0014, 0.0005, 0.0010, 0.0014, 0.0016, 0.0004, 0.0005, 0.0191, 0.0017, 0.0014, 0.0001, 0.0005, 0.0012],
              [0.0063, 0.0023, 0.0031, 0.0028, 0.0010, 0.0019, 0.0030, 0.0038, 0.0011, 0.0017, 0.0024, 0.0031, 0.0009, 0.0012, 0.0017, 0.0126, 0.0047, 0.0003, 0.0010, 0.0024],
              [0.0037, 0.0018, 0.0022, 0.0019, 0.0009, 0.0014, 0.0020, 0.0022, 0.0007, 0.0027, 0.0033, 0.0023, 0.0010, 0.0012, 0.0014, 0.0047, 0.0125, 0.0003, 0.0009, 0.0036],
              [0.0004, 0.0003, 0.0002, 0.0002, 0.0001, 0.0002, 0.0003, 0.0004, 0.0002, 0.0004, 0.0007, 0.0003, 0.0002, 0.0008, 0.0001, 0.0003, 0.0003, 0.0065, 0.0009, 0.0004],
              [0.0013, 0.0009, 0.0007, 0.0006, 0.0003, 0.0007, 0.0009, 0.0008, 0.0015, 0.0014, 0.0022, 0.0010, 0.0006, 0.0042, 0.0005, 0.0010, 0.0009, 0.0009, 0.0102, 0.0015],
              [0.0051, 0.0016, 0.0012, 0.0013, 0.0014, 0.0012, 0.0017, 0.0018, 0.0006, 0.0120, 0.0095, 0.0019, 0.0023, 0.0026, 0.0012, 0.0024, 0.0036, 0.0004, 0.0015, 0.0196]]
    matrix = np.array(matrix)
    diag = [matrix[i,i] for i in range(len(matrix))]
    diag = diag / sum(diag) * len(diag)
    subst_matrix = matrix
    for i in range(len(subst_matrix)):
        subst_matrix[i, i] = 0
    for i in range(len(subst_matrix)):
        subst_matrix[i, :] /= sum(subst_matrix[i, :])
    return aas_dict, subst_matrix, diag


def blosum62_argmaxs():
    aas = 'ARNDCQEGHILKMFPSTWYVBZXJ'
    matrix = [
        [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0,  0],
        [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -1],
        [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -1],
        [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -1],
        [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -2],
        [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -1],
        [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -1],
        [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -1],
        [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -1],
        [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -1],
        [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -1],
        [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -1],
        [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -1],
        [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -1],
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -2],
        [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0,  0],
        [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0,  0],
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -2],
        [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -1],
        [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -1],
        [-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -1],
        [-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -1],
        [ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -1],
        [ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -1]]
    matrix = np.array(matrix)
    blosum_dict = {}
    for i, aa in enumerate(aas):
        sub_dict = {aa_b: matrix[i,j] for j, aa_b in enumerate(aas)}
        blosum_dict[aa] = sub_dict

    blosum_argmaxs = {}
    for aa in aas:
        if aa not in 'BJZX':
            aa_dict = blosum_dict[aa]
            del aa_dict[aa]
            del aa_dict['B']
            del aa_dict['J']
            del aa_dict['Z']
            del aa_dict['X']
            values = np.array(list(aa_dict.values()))
            keys   = np.array(list(aa_dict.keys()))
            amaxs = np.argwhere(values == np.amax(values))
            substs = [keys[a][0] for a in amaxs]
            blosum_argmaxs[aa] = substs
    return blosum_argmaxs
