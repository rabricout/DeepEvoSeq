# C'est notre tête de réseau qui remplace le structure block
# TODO : enlever plein de choses et mettre quelques couches simples à la place. S'inspirer plutôt de la parti RSA de l'evoformer pour ça. 

import math
import sys
import torch
import torch.nn as nn
import numpy as np
from openfold.model.primitives import Linear, LayerNorm


def make_grouped_tensor(m, anchor, human, mouse, max_len, seq_len_crop, residue_index, residue_index_rev, subst_rate=None, subst_pos=None):
    # m is the ouput of the previous part of the network
    batch_size = m.size()[0]
    
    c_tensor = torch.zeros((batch_size, 1, max_len, m.size()[3])).to(m.device)
    for i, c_i in enumerate(seq_len_crop):    # loop on all sequences in the minibatch
        crop_ones = torch.ones((1, 1, c_i.item(), m.size()[3]))
        c_tensor[i,:,:c_i.item(), :] = crop_ones
    
    m_tensor = torch.zeros((batch_size, m.size()[1], max_len, m.size()[3])).to(m.device)
    m_tensor[:,:,:m.size()[2],:] = m
    
    anchor_tensor = torch.zeros((batch_size, 1, max_len, m.size()[3])).to(anchor.device)
    anchor = torch.nn.functional.one_hot(anchor.long(), num_classes=22)
    anchor = anchor[:,None,:,:]
    anchor_tensor[:,:,:m.size()[2],:22] = anchor
    
    human_tensor = torch.zeros((batch_size, 1, max_len, m.size()[3])).to(human.device)
    human = torch.nn.functional.one_hot(human.long(), num_classes=22)
    human = human[:,None,:,:]
    human_tensor[:,:,:m.size()[2],:22] = human
    
    mouse_tensor = torch.zeros((batch_size, 1, max_len, m.size()[3])).to(mouse.device)
    mouse = torch.nn.functional.one_hot(mouse.long(), num_classes=22)
    mouse = mouse[:,None,:,:]
    mouse_tensor[:,:,:m.size()[2],:22] = mouse
    
    r_tensor = torch.zeros((batch_size, 1, m.size()[2], m.size()[3])).to(human.device)
    r1 = residue_index[:,None,:]
    r2 = residue_index_rev[:,None,:]
    r_tensor[:,:,:,0] = r1
    r_tensor[:,:,:,1] = r2

    if subst_rate is not None:
        subst_rate_tensor = torch.zeros((batch_size, 1, max_len, m.size()[3])).to(subst_rate.device)
        subst_rate_tensor[:,:,0,0] = subst_rate
        m = torch.cat((anchor_tensor, human_tensor, mouse_tensor, subst_rate_tensor, m_tensor, c_tensor, r_tensor), dim=1)
        return m

    if subst_pos is not None:
        subst_pos_tensor = torch.zeros((batch_size, 1, max_len, m.size()[3])).to(subst_pos.device)
        subst_pos_tensor[:,0,:subst_pos.size()[1],:2] = subst_pos
        m = torch.cat((anchor_tensor, human_tensor, mouse_tensor, subst_pos_tensor, m_tensor, c_tensor, r_tensor), dim=1)
        return m

    #m = torch.cat((a_tensor, h_tensor, m_tensor, c_tensor, r_tensor), dim=1)
    return None #m


class SubstRateHead(nn.Module):
    """
    Notre tête de réseau qui prédit uniquement le taux de substitution
    """
    def __init__(self, c_m, max_len, transition_n, msa_dim, intern_dim_1, activation, norm, dropout):
        super(SubstRateHead, self).__init__()
        self.max_len = max_len
        self.c_m = c_m
        self.n = transition_n
        self.msa_dim = msa_dim+6#+128
        self.intern_dim_1 = intern_dim_1
        self.dropout = nn.Dropout(p=dropout)

        if activation == 'ReLU':
            self.activ = nn.ReLU()
        elif activation == 'Tanh':
            self.activ = nn.Tanh()
        else:
            self.activ = nn.GELU()
        if norm == 'None':
            self.norm_1 = lambda x: x
        else:
            self.norm_1 = LayerNorm(self.intern_dim_1)
            
        self.linear_rate_1 = Linear(self.c_m, self.intern_dim_1)
        self.linear_rate_2 = Linear(self.intern_dim_1, 8)
        self.linear_rate_3 = Linear(self.msa_dim, 8)
        self.linear_rate_4 = Linear(self.max_len, 32)
        self.linear_rate_flatten_1 = Linear(1+8*8*32, 32)
        self.linear_rate_flatten_2 = Linear(1+32, 1)
        
    def _head(self, m, anchor, human, mouse, seq_len_crop, residue_index, residue_index_rev, subst_rate_HM):
        m_rate = make_grouped_tensor(m, anchor, human, mouse, self.max_len, seq_len_crop, residue_index, residue_index_rev, subst_rate=subst_rate_HM)
        
        m_rate = self.linear_rate_1(m_rate)
        m_rate = self.activ(m_rate)
        m_rate = self.norm_1(m_rate)
        m_rate = self.dropout(m_rate)
        
        m_rate = self.linear_rate_2(m_rate)
        m_rate = self.activ(m_rate)
        m_rate = m_rate.transpose(-1, -3)
        m_rate = self.dropout(m_rate)
        m_rate = self.linear_rate_3(m_rate)
        m_rate = self.activ(m_rate)
        m_rate = m_rate.transpose(-1, -2)
        m_rate = self.dropout(m_rate)
        m_rate = self.linear_rate_4(m_rate)
        m_rate = self.activ(m_rate)
        m_rate = torch.flatten(m_rate, start_dim=1)
    
        #subst_rate_HM = torch.tensor([subst_rate_HM]).to(m_rate.device)    # is a good proxy for the subst_rate_CM
        m_rate = torch.cat((subst_rate_HM, m_rate), dim=1)
        m_rate = self.dropout(m_rate)
        m_rate = self.linear_rate_flatten_1(m_rate)
        m_rate = self.activ(m_rate)
        m_rate = torch.cat((subst_rate_HM, m_rate), dim=1)
        m_rate = self.dropout(m_rate)
        m_rate = self.linear_rate_flatten_2(m_rate)
        return m_rate
        
    def forward(self, m: torch.Tensor, anchor: torch.Tensor, human: torch.Tensor, mouse: torch.Tensor, seq_len_crop, residue_index: torch.tensor, residue_index_rev: torch.tensor, subst_rate_HM) -> torch.Tensor:
        m_rate = self._head(m, anchor, human, mouse, seq_len_crop, residue_index, residue_index_rev, subst_rate_HM[:,None])
        return m_rate
    

    
class SubstPosHead(nn.Module):
    """
    Notre tête de réseau qui prédit les positions des substitutions
    """
    def __init__(self, c_m, max_len, transition_n, msa_dim, intern_dim_1, intern_dim_2, activation, norm, dropout):
        super(SubstPosHead, self).__init__()
        self.max_len = max_len
        self.c_m = c_m
        self.n = transition_n
        self.msa_dim = msa_dim+6
        self.intern_dim_1 = intern_dim_1
        self.intern_dim_2 = intern_dim_2
        self.dropout = nn.Dropout(p=dropout)

        if activation == 'ReLU':
            self.activ = nn.ReLU()
        elif activation == 'Tanh':
            self.activ = nn.Tanh()
        else:
            self.activ = nn.GELU()
        if norm == 'None':
            self.norm_1 = lambda x: x
            self.norm_2 = lambda x: x
        else:
            self.norm_1 = LayerNorm(self.intern_dim_1)
            self.norm_2 = LayerNorm(self.intern_dim_2)
            
        self.linear_1 = Linear(self.c_m, self.intern_dim_1)
        self.linear_2 = Linear(self.intern_dim_1, self.intern_dim_2)
        self.linear_3 = Linear(self.intern_dim_2, self.intern_dim_2)
        self.linear_4 = Linear(self.msa_dim, self.intern_dim_2)
        self.linear_5 = Linear(self.intern_dim_2, 1)
        #self.linear_6 = Linear(self.msa_dim, self.msa_dim, init="relu")
        self.linear_7 = Linear(self.intern_dim_2, 2)
        
    def _head(self, m, anchor, human, mouse, seq_len_crop, residue_index, residue_index_rev, subst_rate):
        m = make_grouped_tensor(m, anchor, human, mouse, self.max_len, seq_len_crop, residue_index, residue_index_rev, subst_rate=subst_rate)
        m = self.linear_1(m)
        m = self.activ(m)
        m = self.norm_1(m)
        m = self.dropout(m)
        
        m = self.linear_2(m)
        m = self.activ(m)
        m = self.norm_2(m)
        m = self.dropout(m)
        
        m = self.linear_3(m)
        m = self.activ(m)
        m = self.norm_2(m)
        m = self.dropout(m)

        # Shape is currently [*, MSA_dim, seq_len, intern_dim_2]
        m = m.transpose(-1, -3)

        #anchor_oneHot = torch.nn.functional.one_hot(anchor.long(), num_classes=22)[:,None,:,:]
        #human_oneHot = torch.nn.functional.one_hot(human.long(), num_classes=22)[:,None,:,:]
        #mouse_oneHot = torch.nn.functional.one_hot(mouse.long(), num_classes=22)[:,None,:,:]
        #print(m.size())
        #print(anchor_oneHot.size())
        #input()
        #m = torch.cat((m, anchor_oneHot, human_oneHot, mouse_oneHot), dim=3)    # skip connexion again with the human, the mouse and the positions of substitution
        
        m = self.linear_4(m)
        m = self.activ(m)
        m = self.norm_2(m)
        m = self.dropout(m)

        m = self.linear_5(m)
        m = self.activ(m)
        m = self.dropout(m)

        # Shape is currently [*, intern_dim_2, seq_len, 1]
        m = m.transpose(-1, -3)

        # Shape is currently [*, 1, seq_len, intern_dim_2]
        #m = self.linear_6(m)
        #m = self.activ(m)

        m = self.linear_7(m)
        # Shape is currently [*, 1, seq_len, 2]
        return m[:,0]
        
    def forward(self, m: torch.Tensor, anchor: torch.Tensor, human: torch.Tensor, mouse: torch.Tensor, seq_len_crop, residue_index, residue_index_rev, subst_rate: torch.Tensor) -> torch.Tensor:
        m = self._head(m, anchor, human, mouse, seq_len_crop, residue_index, residue_index_rev, subst_rate)
        return m



class SeqHead(nn.Module):
    """
    Notre tête de réseau qui remplace le structure module, avec simplement des couches fully-connected
    """
    def __init__(self, c_m, max_len, transition_n, msa_dim, intern_dim_1, intern_dim_2, activation, norm, dropout):
        super(SeqHead, self).__init__()
        self.c_m = c_m
        self.max_len = max_len
        self.n = transition_n
        self.msa_dim = msa_dim+6#+128
        self.intern_dim_1 = intern_dim_1
        self.intern_dim_2 = intern_dim_2
        self.dropout = nn.Dropout(p=dropout)
        
        # Quelques couches simples pour commencer
        if activation == 'ReLU':
            self.activ = nn.ReLU()
        elif activation == 'Tanh':
            self.activ = nn.Tanh()
        else:
            self.activ = nn.GELU()
            
        if norm == 'None':
            self.norm_1 = lambda x: x
            self.norm_2 = lambda x: x
        else:
            self.norm_1 = LayerNorm(self.intern_dim_1)
            self.norm_2 = LayerNorm(self.intern_dim_2)
    
        self.linear_1 = Linear(self.c_m, self.intern_dim_1)
        self.linear_2 = Linear(self.intern_dim_1, self.intern_dim_1)
        self.linear_3 = Linear(self.intern_dim_1, self.intern_dim_2)
        self.linear_4 = Linear(self.intern_dim_2, self.intern_dim_2)
        self.linear_5 = Linear(self.intern_dim_2, 1)
        self.linear_6 = Linear(self.msa_dim+3*22+2, self.msa_dim)    # add the 3 species in one-hot plus the positions
        self.linear_7 = Linear(self.msa_dim, self.msa_dim)
        self.linear_8 = Linear(self.msa_dim, 22)
        
    def _head(self, m, anchor, human, mouse, seq_len_crop, residue_index, residue_index_rev, subst_pos):
        m = make_grouped_tensor(m, anchor, human, mouse, self.max_len, seq_len_crop, residue_index, residue_index_rev, subst_pos=subst_pos)
        m = self.linear_1(m)
        m = self.activ(m)
        m = self.norm_1(m)
        m = self.dropout(m)
        
        m = self.linear_2(m)
        m = self.activ(m)
        m = self.norm_1(m)
        m = self.dropout(m)
        
        m = self.linear_3(m)
        m = self.activ(m)
        m = self.norm_2(m)
        m = self.dropout(m)
        
        # Shape is currently [1, MSA_dim, seq_len, intern_dim_2]
        m = self.linear_4(m)
        m = self.activ(m)
        m = self.norm_2(m)
        m = self.dropout(m)
        
        m = self.linear_5(m)
        m = self.activ(m)
        m = self.dropout(m)
        
        # Shape is now [*, MSA_dim, seq_len, 1]
        m = m.transpose(-1, -3)

        anchor_oneHot = torch.nn.functional.one_hot(anchor.long(), num_classes=22)[:,None,:,:]
        human_oneHot = torch.nn.functional.one_hot(human.long(), num_classes=22)[:,None,:,:]
        mouse_oneHot = torch.nn.functional.one_hot(mouse.long(), num_classes=22)[:,None,:,:]
        subst_pos = subst_pos[:,None,:,:]
        m = torch.cat((m, anchor_oneHot, human_oneHot, mouse_oneHot, subst_pos), dim=3)    # skip connexion again with the human, the mouse and the positions of substitution
        
        m = self.linear_6(m)
        m = self.activ(m)
        m = self.linear_7(m)
        m = self.activ(m)
        
        m = self.linear_8(m)
        # Shape is now [*, 1, seq_len, 22]
        return m[:,0]
        
    def forward(self, m: torch.Tensor, anchor: torch.Tensor, human: torch.Tensor, mouse: torch.Tensor, seq_len_crop, residue_index, residue_index_rev, subst_pos: torch.Tensor) -> torch.Tensor:
        m = self._head(m, anchor, human, mouse, seq_len_crop, residue_index, residue_index_rev, subst_pos)
        return m

