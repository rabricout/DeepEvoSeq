# C'est notre tête de réseau qui remplace le structure block
# TODO : enlever plein de choses et mettre quelques couches simples à la place. S'inspirer plutôt de la parti RSA de l'evoformer pour ça. 

import time
import math
import sys
import torch
import torch.nn as nn
import numpy as np
from openfold.model.primitives import Linear, LayerNorm
import openfold.np.residue_constants as residue_constants

import random
# To get closest seq
from difflib import get_close_matches
# To get blosum matrix
import openfold.utils.my_functions as my_functions


class BaselinesModule(nn.Module):
    def __init__(self, strat):
        super(BaselinesModule, self).__init__()
        self.transitionMatrix = TransitionMatrixModule()
        self.strat = strat
        self.blosum_argmaxs = my_functions.blosum62_argmaxs()
        self.blosum_dict, self.blosum_subst, self.blosum_diag_norm = my_functions.blosum62_freq()
        
                
    def forward(self, anchor_seq: torch.Tensor, label_seq: torch.Tensor, human_seq: torch.Tensor, mouse_seq: torch.Tensor, msa_feat: torch.Tensor, residue_index: torch.Tensor, residue_index_rev: torch.Tensor, mask_substitutions: torch.Tensor, nature_only, proxy_subst_rate) -> torch.Tensor:
        strat = self.strat
        msa = msa_feat[:, :, :, :22]    # keep only the aa values and not the rest

        anchor_str = residue_constants.aatype_to_str_sequence(anchor_seq[0].int().tolist())
        label_str = residue_constants.aatype_to_str_sequence(label_seq[0].int().tolist())
        human_str = residue_constants.aatype_to_str_sequence(human_seq[0].int().tolist())
        mouse_str = residue_constants.aatype_to_str_sequence(mouse_seq[0].int().tolist())


        if False:
            msa_1_str = residue_constants.onehot_to_str_sequence(np.array(msa[0,0,:,:].cpu()))
            msa_2_str = residue_constants.onehot_to_str_sequence(np.array(msa[0,1,:,:].cpu()))
            msa_10_str = residue_constants.onehot_to_str_sequence(np.array(msa[0,9,:,:].cpu()))
            print(anchor_str)
            print()
            print(label_str)
            print()
            print(human_str)
            print()
            print(mouse_str)
            print()
            print(msa_1_str)
            print()
            print(msa_2_str)
            print()
            print(msa_10_str)
            print()
            input('i my_baselines.py')
        
        #msa_seqs = []

        #for s in range(msa.size()[1]):
        #    msa_seq = [aa_to_id_dict[torch.argmax(i).item()] if torch.max(i)>0 else '-' for i in msa[0,s]]
        #    msa_seq = ''.join(msa_seq)
        #    msa_seqs.append(msa_seq)
        # use anchor_seq and msa_seqs (these are strings) for custom predictions
        
        m_len = len(anchor_str)


        # Construct the majoritatian substitutions
        #subst_rates = []
        #maj_substs = []
        #maj_aa = []
        #for p in range(m_len):
        #    msa_p = ''.join([s[p] for s in msa_seqs]) # it might be full of indels (-)
        #    maj_aa_tmp = max(msa_p, key=lambda x: msa_p.count(x))
        #    maj_aa.append(maj_aa_tmp)
        #    local_subst = msa_p.replace(anchor_str[p], '')
        #    subst_rate = len(local_subst) / m_len
        #    subst_rates.append(subst_rate)
        #    if len(local_subst) > 0:
        #        maj_subst = max(local_subst, key=lambda x: local_subst.count(x))
        #    else:
        #        maj_subst = '-'
        #    maj_substs.append(maj_subst)
        
        #indexes = list(range(m_len))
        #subst_rates_s, maj_substs_s, indexes_s = (list(t) for t in zip(*sorted(zip(subst_rates, maj_substs, indexes), reverse=True)))


        
        #prediction = nn.functional.one_hot(anchor_seq, num_classes=23)
        #prediction = prediction[None, :]

        #zeros = torch.zeros(1, m_len, 1)
        #ones  = torch.ones(1, m_len, 1)
        #position_prediction = torch.cat((ones, zeros), -1).to(prediction.device)

        #subst_pos = []
        #for i in range(len(mask_substitutions[0])):
        #    if mask_substitutions[0,i] == 1:
        #        subst_pos.append(i)
        if nature_only:
            strats = ['mouse', 'human', 'ancestor', 'most_subst', 'ancestor_blosum_argmax', 'ancestor_blosum_argmax', 'ancestorHuman_blosum_argmax', 'ancestorHuman_transitionMatrix_argmax' ,'most_subst_randomized']
            if strat not in strats:
                print("ATTENTION, LA STRATEGIE N'EST PAS BONNE")

            # Building the ancestor seq knowing the substitution mask so the mouse is the outgroup
            ancestor_seq = []
            for i in range(len(anchor_seq[0])):
                if mask_substitutions[0,i] == 0:    # no substitution
                    ancestor_seq.append(anchor_seq[0,i])
                else:
                    if mouse_seq[0,i] == anchor_seq[0,i]:    # substitution in the target branch: the anchor is the ancestor
                        ancestor_seq.append(mouse_seq[0,i])
                    else:    # substitution in the anchor branch: the outgroup (mouse) is the ancestor
                        ancestor_seq.append(mouse_seq[0,i])
            ancestor_seq = torch.tensor(ancestor_seq)[None,:].to(anchor_seq.device)
                
            # Nature only so the predictions of positions are perfect
            prediction_position = nn.functional.one_hot(mask_substitutions, 2)

            # Start with the anchor seq and modify only the positions in the substitution mask
            prediction = nn.functional.one_hot(anchor_seq.long(), num_classes=22)

            
            if strat == 'mouse':
                prediction_aatype = mouse_seq
                for i, v in enumerate(mask_substitutions[0]):
                    if v == 1:    # prediction is anchor by default, and we adapt it only on substitution positions
                        prediction[0,i] = nn.functional.one_hot(prediction_aatype.long()[0,i], num_classes=22)
                
            if strat == 'human':
                prediction_aatype = human_seq
                for i, v in enumerate(mask_substitutions[0]):
                    if v == 1:    # prediction is anchor by default, and we adapt it only on substitution positions
                        prediction[0,i] = nn.functional.one_hot(prediction_aatype.long()[0,i], num_classes=22)
                
            if strat == 'ancestor':
                prediction_aatype = ancestor_seq
                for i, v in enumerate(mask_substitutions[0]):
                    if v == 1:
                        prediction[0,i] = nn.functional.one_hot(prediction_aatype.long()[0,i], num_classes=22)

            if strat == 'ancestor_blosum_argmax':
                for i, v in enumerate(mask_substitutions[0]):
                    if v == 1:
                        if mouse_seq[0,i] == anchor_seq[0,i]:    # substitution in the target branch: the anchor is the ancestor
                            new_aa = random.choice(self.blosum_argmaxs[anchor_str[i]])    # choose an aa from the anchor based on blosum substs
                            t = torch.tensor(residue_constants.str_to_aatype(new_aa)) # use residue_constants
                            hot = nn.functional.one_hot(t, num_classes=22)
                            prediction[:, i] = hot
                        else:    # substitution in the anchor branch: predict the outgroup mouse (probable ancestor)
                            prediction[0,i] = nn.functional.one_hot(mouse_seq.long()[0,i], num_classes=22)
            
            if strat == 'ancestorHuman_blosum_argmax':
                for i, v in enumerate(mask_substitutions[0]):
                    if v == 1:    # substitution
                        if human_seq[0,i] == anchor_seq[0,i]:    # substitution in the target branch: the anchor is the ancestor
                            new_aa = random.choice(self.blosum_argmaxs[anchor_str[i]])    # choose an aa from the anchor based on blosum substs
                            t = torch.tensor(residue_constants.str_to_aatype(new_aa)) # use residue_constants
                            hot = nn.functional.one_hot(t, num_classes=22)
                            prediction[:, i] = hot
                        else:    # substitution in the anchor branch: predict the outgroup human (probable ancestor)
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
            
            if strat == 'ancestor_transitionMatrix_argmax':
                for i, v in enumerate(mask_substitutions[0]):
                    if v == 1:
                        if mouse_seq[0,i] == anchor_seq[0,i]:    # substitution in the target branch: the anchor is the ancestor
                            new_aa = self.transitionMatrix.predict_aa(anchor_seq[0,i].long().cpu())    # aa is not in str but aatype
                            #new_aa = random.choice(self.blosum_argmaxs[anchor_str[i]])    # choose an aa from the anchor based on blosum substs
                            #t = torch.tensor(residue_constants.str_to_aatype([new_aa])) # use residue_constants
                            hot = nn.functional.one_hot(new_aa, num_classes=22)
                            prediction[:, i] = hot
                        else:    # substitution in the anchor branch: predict the outgroup mouse (probable ancestor)
                            prediction[0,i] = nn.functional.one_hot(mouse_seq.long()[0,i], num_classes=22)
                self.transitionMatrix.updateTransitionMatrix(label_seq[0].long().cpu(), anchor_seq[0].long().cpu())
                
            if strat == 'ancestorHuman_transitionMatrix_argmax':
                for i, v in enumerate(mask_substitutions[0]):
                    if v == 1:
                        if human_seq[0,i] == anchor_seq[0,i]:    # substitution in the target branch: the anchor is the ancestor
                            new_aa = self.transitionMatrix.predict_aa(anchor_seq[0,i].long().cpu())    # aa is not in str but aatype
                            #new_aa = random.choice(self.blosum_argmaxs[anchor_str[i]])    # choose an aa from the anchor based on blosum substs
                            #t = torch.tensor(residue_constants.str_to_aatype([new_aa])) # use residue_constants
                            hot = nn.functional.one_hot(new_aa, num_classes=22)
                            prediction[:, i] = hot
                        else:    # substitution in the anchor branch: predict the outgroup mouse (probable ancestor)
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
                self.transitionMatrix.updateTransitionMatrix(label_seq[0].long().cpu(), anchor_seq[0].long().cpu())
            
            if strat == 'most_subst':
                for i in subst_pos:
                    if anchor_str[i] != '-':
                        t = torch.Tensor([inv_dict[maj_substs[i]]]).long()
                        hot = nn.functional.one_hot(t, num_classes=23)
                        prediction[:, i] = hot
                        position_prediction[:, i] = torch.tensor([0,1])
            if strat == 'most_subst_randomized':
                #c_random = 0
                for i in subst_pos:
                    if anchor_str[i] != '-':
                        if maj_substs[i]=='-':
                            t = torch.Tensor([np.random.randint(0,21)]).long()
                            #c_random += 1
                        else:
                            t = torch.Tensor([inv_dict[maj_substs[i]]]).long()
                        hot = nn.functional.one_hot(t, num_classes=23)
                        prediction[:, i] = hot
                        position_prediction[:, i] = torch.tensor([0,1])

        else:    # Positions are not known
            strats = {'mouse', 'human', 'ancestor', 'ancestorHuman'}
            if strat not in strats:
                print("ATTENTION, LA STRATEGIE N'EST PAS BONNE")
                
            if strat == 'mouse':
                prediction_seq = mouse_seq    # copy the mouse
                prediction = nn.functional.one_hot(prediction_seq.long(), num_classes=22)
                prediction_position_mask = torch.zeros((1, m_len))    # for the position, start with no substitution and add the mouse substitutions
                for i, v in enumerate(mouse_seq[0]):
                    if prediction_seq[0,i] != anchor_seq[0,i]:
                        prediction_position_mask[0,i] = 1
                prediction_position = nn.functional.one_hot(prediction_position_mask.long(), 2)
            
            if strat == 'human':
                prediction_seq = human_seq    # copy the human
                prediction = nn.functional.one_hot(prediction_seq.long(), num_classes=22)
                prediction_position_mask = torch.zeros((1, m_len))    # for the position, start with no substitution and add the mouse substitutions
                for i, v in enumerate(human_seq[0]):
                    if prediction_seq[0,i] != anchor_seq[0,i]:
                        prediction_position_mask[0,i] = 1
                prediction_position = nn.functional.one_hot(prediction_position_mask.long(), 2)
            
            if strat == 'ancestor':
                # construct ancestral sequence, not knowing the positions of substitutions, so mouse is the outgroup
                ancestor_seq = []
                for i in range(len(anchor_seq[0])):
                    if anchor_seq[0,i] == human_seq[0,i]:    # no substitution between anchor and human
                        ancestor_seq.append(anchor_seq[0,i])
                    elif mouse_seq[0,i] == human_seq[0,i] or mouse_seq[0,i] == anchor_seq[0,i]:    # substitution, so we get the outgroup if it is similar to one of the two species
                        ancestor_seq.append(mouse_seq[0,i])
                    else:    # 3 species are different, take the closest one
                        ancestor_seq.append(anchor_seq[0,i])
                ancestor_seq = torch.tensor(ancestor_seq)[None,:].to(anchor_seq.device)

                prediction_seq = ancestor_seq    # copy the ancestor
                prediction = nn.functional.one_hot(prediction_seq.long(), num_classes=22)
                prediction_position_mask = torch.zeros((1, m_len))    # for the position, start with no substitution and add the ancestor substitutions
                for i, v in enumerate(mouse_seq[0]):
                    if prediction_seq[0,i] != anchor_seq[0,i]:
                        prediction_position_mask[0,i] = 1
                prediction_position = nn.functional.one_hot(prediction_position_mask.long(), 2)
            
            if strat == 'ancestorHuman':
                # construct ancestral sequence, not knowing the positions of substitutions, so human is the outgroup
                ancestor_seq = []
                for i in range(len(anchor_seq[0])):
                    if anchor_seq[0,i] == mouse_seq[0,i]:    # no substitution between anchor and mouse
                        ancestor_seq.append(anchor_seq[0,i])
                    elif human_seq[0,i] == mouse_seq[0,i] or human_seq[0,i] == anchor_seq[0,i]:    # substitution, so we get the outgroup if it is similar to one of the two species
                        ancestor_seq.append(human_seq[0,i])
                    else:    # 3 species are different, take the closest one
                        ancestor_seq.append(anchor_seq[0,i])
                ancestor_seq = torch.tensor(ancestor_seq)[None,:].to(anchor_seq.device)

                prediction_seq = ancestor_seq    # copy the ancestor
                prediction = nn.functional.one_hot(prediction_seq.long(), num_classes=22)
                prediction_position_mask = torch.zeros((1, m_len))    # for the position, start with no substitution and add the ancestor substitutions
                for i, v in enumerate(mouse_seq[0]):
                    if prediction_seq[0,i] != anchor_seq[0,i]:
                        prediction_position_mask[0,i] = 1
                prediction_position = nn.functional.one_hot(prediction_position_mask.long(), 2)
            
            if strat == 'closest':    # just copy the closest sequence in the alignment
                closest_str = get_close_matches(anchor_str, msa_seqs, n=1, cutoff=0.1)
                if len(closest_str) > 0:
                    closest_str = closest_str[0]
                else:
                    closest_str = input_str
                for i in range(m_len):
                    #if closest_str[i] != '-':
                    t = torch.Tensor([inv_dict[closest_str[i]]]).long()
                    hot = nn.functional.one_hot(t, num_classes=23)
                    prediction[:, i] = hot
                    if anchor_str[i] != closest_str[i]:
                        position_prediction[:, i] = torch.tensor([0,1])      

            if strat == 'maj':    # for each position we predict the most abundant one if it's not an indel
                for i in range(m_len):
                    if maj_aa[i] != '-':
                        t = torch.Tensor([inv_dict[maj_aa[i]]]).long()
                        hot = nn.functional.one_hot(t, num_classes=23)
                        prediction[:, i] = hot
                        if anchor_str[i] != maj_aa[i]:
                            position_prediction[:, i] = torch.tensor([0,1])
    
            if strat == 'most_subst':    # we predit a substitution for the most changing positions (positions with most aa different than anchor). If so, we choose the most abundant substitution at this position
                #subst_thresh = 0.1 #proxy_subst_rate #0.1
                subst_thresh = proxy_subst_rate
                for i in range(m_len_no_indels):
                    if i < int(m_len_no_indels*subst_thresh):
                        t = torch.Tensor([inv_dict[maj_substs_s[i]]]).long()
                        hot = nn.functional.one_hot(t, num_classes=23)
                        prediction[:, indexes_s[i]] = hot
                        position_prediction[:, indexes_s[i]] = torch.tensor([0,1])

        # consider them as tensors with grad only for the pipeline (even if it's ugly)
        prediction_position = prediction_position.float().to(anchor_seq.device)
        prediction = prediction.float().to(anchor_seq.device)
        prediction_position.requires_grad = True
        prediction.requires_grad = True
        return prediction_position, prediction



class TransitionMatrixModule():
    def __init__(self):
        self.transitionMatrix = torch.zeros((22,22))

    def updateTransitionMatrix(self, target_seq: torch.Tensor, anchor_seq: torch.Tensor):
        for i in range(target_seq.size()[0]):
            if anchor_seq[i] != target_seq[i]:
                self.transitionMatrix[anchor_seq[i], target_seq[i]] += 1

    def predict_aa(self, anchor_aa):
        return torch.argmax(self.transitionMatrix[anchor_aa])
