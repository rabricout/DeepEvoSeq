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
        self.random_transition_module = TransitionRandomModule()
        
                
    def forward(self, anchor_seq: torch.Tensor, label_seq: torch.Tensor, human_seq: torch.Tensor, mouse_seq: torch.Tensor, msa_feat: torch.Tensor, residue_index: torch.Tensor, residue_index_rev: torch.Tensor, merged_mask: torch.Tensor, mask_substitutions_ancestor: torch.Tensor, nature_only, proxy_subst_rate, training) -> torch.Tensor:
        strat = self.strat
        msa = msa_feat[:, :, :, :22]    # keep only the aa values and not the rest
        msa_seqs = [residue_constants.onehot_to_str_sequence(np.array(msa[0,i,:,:].cpu())) for i in range(msa.size()[1])]

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
            strats = ['closest', 'random', 'mouse', 'human', 'blosum_argmax', 'transitionMatrix_argmax']
            if strat not in strats:
                print("ATTENTION, LA STRATEGIE N'EST PAS BONNE")

            # Building the ancestor seq knowing the substitution mask so the mouse is the outgroup
            ancestor_seq = []
            for i in range(len(anchor_seq[0])):
                if mask_substitutions_ancestor[0,i] == 0:    # no substitution
                    ancestor_seq.append(anchor_seq[0,i])
                else:
                    if mouse_seq[0,i] == anchor_seq[0,i]:    # substitution in the target branch: the anchor is the ancestor
                        ancestor_seq.append(mouse_seq[0,i])
                    else:    # substitution in the anchor branch: the outgroup (mouse) is the ancestor
                        ancestor_seq.append(mouse_seq[0,i])
            ancestor_seq = torch.tensor(ancestor_seq)[None,:].to(anchor_seq.device)
            
            # Nature only so the predictions of positions are perfect
            prediction_position = nn.functional.one_hot(mask_substitutions_ancestor, 2)

            # Start with the anchor seq and modify only the positions in the substitution mask
            prediction = nn.functional.one_hot(anchor_seq.long(), num_classes=22)

            if strat == 'closest':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:    # prediction is anchor by default, and we adapt it only on substitution positions
                        if human_seq[0,i] != anchor_seq[0,i]:
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
                        elif mouse_seq[0,i] != anchor_seq[0,i]:
                            prediction[0,i] = nn.functional.one_hot(mouse_seq.long()[0,i], num_classes=22)
                        elif msa_1_seq[0,i] != anchor_seq[0,i]:
                            prediction[0,i] = nn.functional.one_hot(msa_1_seq.long()[0,i], num_classes=22)
                        else:
                            new_aa = random.choice(self.blosum_argmaxs[anchor_str[i]])    # choose an aa from the anchor based on blosum substs
                            t = torch.tensor(residue_constants.str_to_aatype(new_aa)) # use residue_constants
                            hot = nn.functional.one_hot(t, num_classes=22)
                            prediction[0,i] = hot
                        
            if strat == 'random':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:    # prediction is anchor by default, and we adapt it only on substitution positions
                        new_aa = self.random_transition_module.predict_aa(anchor_str[i])
                        t = torch.tensor(residue_constants.str_to_aatype(new_aa)) # use residue_constants
                        hot = nn.functional.one_hot(t, num_classes=22)
                        prediction[:, i] = hot
            
            if strat == 'mouse':
                prediction_aatype = mouse_seq
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:    # prediction is anchor by default, and we adapt it only on substitution positions
                        prediction[0,i] = nn.functional.one_hot(prediction_aatype.long()[0,i], num_classes=22)
                
            if strat == 'human':
                prediction_aatype = human_seq
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:    # prediction is anchor by default, and we adapt it only on substitution positions
                        prediction[0,i] = nn.functional.one_hot(prediction_aatype.long()[0,i], num_classes=22)

            if strat == 'transitionMatrix_argmax':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:
                            new_aa = self.transitionMatrix.predict_aa(anchor_seq[0,i].long().cpu())    # aa is not in str but aatype
                            hot = nn.functional.one_hot(new_aa, num_classes=22)
                            prediction[:, i] = hot
                if training:
                    self.transitionMatrix.updateTransitionMatrix(label_seq[0].long().cpu(), anchor_seq[0].long().cpu())
                    self.transitionMatrix.print_transitionMatrix()

            if strat == 'blosum_argmax':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:    # substitution
                        new_aa = random.choice(self.blosum_argmaxs[anchor_str[i]])    # choose an aa from the anchor based on blosum substs
                        t = torch.tensor(residue_constants.str_to_aatype(new_aa)) # use residue_constants
                        hot = nn.functional.one_hot(t, num_classes=22)
                        prediction[:,i] = hot
                        
            if strat == 'biancestor':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:
                        if human_seq[0,i] == mouse_seq[0,i] and human_seq[0,i] != anchor_seq[0,i]:    # biancestor case, choose the human/mouse
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
                        elif human_seq[0,i] != anchor_seq[0,i]:
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
                        elif mouse_seq[0,i] != anchor_seq[0,i]:
                            prediction[0,i] = nn.functional.one_hot(mouse_seq.long()[0,i], num_classes=22)
                        else:    # all same
                            prediction[0,i] = nn.functional.one_hot(anchor_seq.long()[0,i], num_classes=22)
                        
            if strat == 'ancestor':
                prediction_aatype = ancestor_seq
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:
                        prediction[0,i] = nn.functional.one_hot(prediction_aatype.long()[0,i], num_classes=22)

            if strat == 'ancestor_blosum_argmax':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:
                        if mouse_seq[0,i] == anchor_seq[0,i]:    # substitution in the target branch: the anchor is the ancestor
                            new_aa = random.choice(self.blosum_argmaxs[anchor_str[i]])    # choose an aa from the anchor based on blosum substs
                            t = torch.tensor(residue_constants.str_to_aatype(new_aa)) # use residue_constants
                            hot = nn.functional.one_hot(t, num_classes=22)
                            prediction[:,i] = hot
                        else:    # substitution in the anchor branch: predict the outgroup mouse (probable ancestor)
                            prediction[0,i] = nn.functional.one_hot(mouse_seq.long()[0,i], num_classes=22)
            
            if strat == 'ancestorHuman_blosum_argmax':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:    # substitution
                        if human_seq[0,i] == anchor_seq[0,i]:    # substitution in the target branch: the anchor is the ancestor
                            new_aa = random.choice(self.blosum_argmaxs[anchor_str[i]])    # choose an aa from the anchor based on blosum substs
                            t = torch.tensor(residue_constants.str_to_aatype(new_aa)) # use residue_constants
                            hot = nn.functional.one_hot(t, num_classes=22)
                            prediction[:,i] = hot
                        else:    # substitution in the anchor branch: predict the outgroup human (probable ancestor)
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
            
            if strat == 'ancestor_transitionMatrix_argmax':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:
                        if mouse_seq[0,i] == anchor_seq[0,i]:    # substitution in the target branch: the anchor is the ancestor
                            new_aa = self.transitionMatrix.predict_aa(anchor_seq[0,i].long().cpu())    # aa is not in str but aatype
                            hot = nn.functional.one_hot(new_aa, num_classes=22)
                            prediction[:, i] = hot
                        else:    # substitution in the anchor branch: predict the outgroup mouse (probable ancestor)
                            prediction[0,i] = nn.functional.one_hot(mouse_seq.long()[0,i], num_classes=22)
                if training:
                    self.transitionMatrix.updateTransitionMatrix(label_seq[0].long().cpu(), anchor_seq[0].long().cpu())
                
            if strat == 'ancestorHuman_transitionMatrix_argmax':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:
                        if human_seq[0,i] == anchor_seq[0,i]:    # substitution in the target branch: the anchor is the ancestor
                            new_aa = self.transitionMatrix.predict_aa(anchor_seq[0,i].long().cpu())    # aa is not in str but aatype
                            hot = nn.functional.one_hot(new_aa, num_classes=22)
                            prediction[:, i] = hot
                        else:    # substitution in the anchor branch: predict the outgroup human (probable ancestor)
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
                if training:
                    self.transitionMatrix.updateTransitionMatrix(label_seq[0].long().cpu(), anchor_seq[0].long().cpu())
                    self.transitionMatrix.print_transitionMatrix()

            if strat == 'biancestor_transitionMatrix_argmax':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:
                        if human_seq[0,i] == mouse_seq[0,i] and human_seq[0,i] != anchor_seq[0,i]:    # biancestor case, choose the human/mouse
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
                        elif human_seq[0,i] != anchor_seq[0,i]:
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
                        elif mouse_seq[0,i] != anchor_seq[0,i]:
                            prediction[0,i] = nn.functional.one_hot(mouse_seq.long()[0,i], num_classes=22)
                        else:    # all different
                            new_aa = self.transitionMatrix.predict_aa(anchor_seq[0,i].long().cpu())    # aa is not in str but aatype
                            hot = nn.functional.one_hot(new_aa, num_classes=22)
                            prediction[:, i] = hot
                if training:
                    self.transitionMatrix.updateTransitionMatrix(label_seq[0].long().cpu(), anchor_seq[0].long().cpu())

            if strat == 'biancestor_BLOSUM_argmax':
                for i, v in enumerate(mask_substitutions_ancestor[0]):
                    if v == 1:
                        if human_seq[0,i] == mouse_seq[0,i] and human_seq[0,i] != anchor_seq[0,i]:    # biancestor case, choose the human/mouse
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
                        elif human_seq[0,i] != anchor_seq[0,i]:
                            prediction[0,i] = nn.functional.one_hot(human_seq.long()[0,i], num_classes=22)
                        elif mouse_seq[0,i] != anchor_seq[0,i]:
                            prediction[0,i] = nn.functional.one_hot(mouse_seq.long()[0,i], num_classes=22)
                        else:    # all different
                            new_aa = random.choice(self.blosum_argmaxs[anchor_str[i]])    # choose an aa from the anchor based on blosum substs
                            t = torch.tensor(residue_constants.str_to_aatype(new_aa)) # use residue_constants
                            hot = nn.functional.one_hot(t, num_classes=22)
                            prediction[:,i] = hot
                if training:
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
            strats = {'random', 'mouse', 'human', 'proxy'}    #, 'ancestor', 'ancestorHuman', 'biancestor', 'mouse_adaptative', 'human_adaptative'}
            if strat not in strats:
                print("ATTENTION, LA STRATEGIE N'EST PAS BONNE")

            if strat == 'random':
                rate = proxy_subst_rate
                prediction = nn.functional.one_hot(anchor_seq.long(), num_classes=22)
                prediction_position_mask = torch.zeros((1, m_len))    # for the position, start with no substitution and add the mouse substitutions
                for i, v in enumerate(prediction[0]):
                    if random.random() < rate:
                        new_aa = self.transitionMatrix.predict_aa(anchor_seq[0,i].long().cpu())    # aa is not in str but aatype
                        hot = nn.functional.one_hot(new_aa, num_classes=22)
                        prediction[:, i] = hot
                        prediction_position_mask[0,i] = 1
                prediction_position = nn.functional.one_hot(prediction_position_mask.long(), 2)
                
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

            if strat == 'proxy':
                #TODO : count number of different aa for each position, get the most probable ones
                prediction = nn.functional.one_hot(anchor_seq.long(), num_classes=22)
                prediction_position_mask = torch.zeros((1, m_len))    # for the position, start with no substitution and add the mouse substitutions
                aa_stability = {}
                maj_subst = []
                for p in range(m_len):
                    #msa_p = ''.join([s[p] for s in msa_seqs]) # it might be full of indels (-)
                    #msa_p += anchor_str[p] + mouse_str[p] + human_str[p]
                    msa_p = anchor_str[p] + mouse_str[p] + human_str[p]
                    #msa_p = msa_p.replace('-', '')
                    nb_letters = len(set(msa_p))
                    if nb_letters in aa_stability.keys():
                        aa_stability[nb_letters].append(p)
                    else:
                        aa_stability[nb_letters] = [p]
                    local_subst = msa_p.replace(anchor_str[p], '')
                    if len(local_subst) > 0:
                        maj_subst.append(max(local_subst, key=lambda x: local_subst.count(x)))
                    else:
                        maj_subst.append('-')
                nb_subst = proxy_subst_rate*sum(merged_mask[0])
                c_subst = 0
                nb_max = max(aa_stability.keys())
                for i in range(nb_max, 0, -1):
                    if i in aa_stability.keys() and merged_mask[0,i] == 1:
                        for j in range(len(aa_stability[i])):
                            if j%2==0:    # get extremities first
                                idx = aa_stability[i][j//2]
                            else:
                                idx = aa_stability[i][-(j+1)//2]
                            if c_subst < nb_subst:
                                c_subst += 1
                                prediction_position_mask[0,idx] = 1
                                aa_pred = residue_constants.str_to_aatype(np.array([maj_subst[idx]]))
                                prediction[0,i] = nn.functional.one_hot(torch.tensor(aa_pred), num_classes=22)
                prediction_position = nn.functional.one_hot(prediction_position_mask.long(), 2)

                
                
            if strat == 'ancestor':
                # construct ancestral sequence, not knowing the positions of substitutions, so mouse is the outgroup
                ancestor_seq = []
                for i in range(len(anchor_seq[0])):
                    if anchor_seq[0,i] == human_seq[0,i]:    # no substitution between anchor and human
                        ancestor_seq.append(anchor_seq[0,i])
                    elif mouse_seq[0,i] == human_seq[0,i] or mouse_seq[0,i] == anchor_seq[0,i]:    # substitution, so we get the outgroup if it is similar to one of the two species
                        ancestor_seq.append(mouse_seq[0,i])
                    else:    # 3 species are different, predict a substitution
                        ancestor_seq.append(mouse_seq[0,i])
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
                    else:    # 3 species are different, predict a substitution
                        #ancestor_seq.append(anchor_seq[0,i])
                        ancestor_seq.append(human_seq[0,i])
                ancestor_seq = torch.tensor(ancestor_seq)[None,:].to(anchor_seq.device)

                prediction_seq = ancestor_seq    # copy the ancestor
                prediction = nn.functional.one_hot(prediction_seq.long(), num_classes=22)
                prediction_position_mask = torch.zeros((1, m_len))    # for the position, start with no substitution and add the ancestor substitutions
                for i, v in enumerate(mouse_seq[0]):
                    if prediction_seq[0,i] != anchor_seq[0,i]:
                        prediction_position_mask[0,i] = 1
                prediction_position = nn.functional.one_hot(prediction_position_mask.long(), 2)
                
            if strat == 'biancestor':
                # construct ancestral sequence, not knowing the positions of substitutions, so human is the outgroup
                ancestor_seq = []
                for i in range(len(anchor_seq[0])):
                    if anchor_seq[0,i] == mouse_seq[0,i] and anchor_seq[0,i] == human_seq[0,i]:    # no substitution between anchor and mouse
                        ancestor_seq.append(anchor_seq[0,i])
                    elif human_seq[0,i] == anchor_seq[0,i]:    # substitution
                        ancestor_seq.append(human_seq[0,i])
                    elif mouse_seq[0,i] == anchor_seq[0,i]:    # substitution
                        ancestor_seq.append(mouse_seq[0,i])
                    elif mouse_seq[0,i] == human_seq[0,i]:    # substitution
                        ancestor_seq.append(mouse_seq[0,i])
                    else:    # the 3 species are different
                        ancestor_seq.append(anchor_seq[0,i])
                ancestor_seq = torch.tensor(ancestor_seq)[None,:].to(anchor_seq.device)

                prediction_seq = ancestor_seq    # copy the ancestor
                prediction = nn.functional.one_hot(prediction_seq.long(), num_classes=22)
                prediction_position_mask = torch.zeros((1, m_len))    # for the position, start with no substitution and add the ancestor substitutions
                for i, v in enumerate(mouse_seq[0]):
                    if prediction_seq[0,i] != anchor_seq[0,i]:
                        prediction_position_mask[0,i] = 1
                prediction_position = nn.functional.one_hot(prediction_position_mask.long(), 2)

            if strat == 'mouse_adaptative':
                prediction_seq = mouse_seq    # copy the mouse
                prediction = nn.functional.one_hot(prediction_seq.long(), num_classes=22)
                prediction_position_mask = torch.zeros((1, m_len))    # for the position, start with no substitution and add the mouse substitutions
                subst_positions = []
                tot_positions = 0    # counting positions without gap for the subst rate
                for i, v in enumerate(mouse_seq[0]):
                    if prediction_seq[0,i] != 21 and anchor_seq[0,i] != 21:
                        tot_positions += 1
                        if prediction_seq[0,i] != anchor_seq[0,i]:
                            subst_positions.append(i)
                if tot_positions > 0:
                    mouse_subst_rate = len(subst_positions) / tot_positions
                    if mouse_subst_rate > proxy_subst_rate:    # in this case, reduce the number of substitutions in the prediction
                        subst_nb = int(proxy_subst_rate*tot_positions)
                        subst_positions = random.choices(subst_positions, k=subst_nb)
                    for p in np.array(subst_positions):
                        prediction_position_mask[0,p] = 1
                prediction_position = nn.functional.one_hot(prediction_position_mask.long(), 2)
                
            if strat == 'human_adaptative':
                prediction_seq = human_seq    # copy the human
                prediction = nn.functional.one_hot(prediction_seq.long(), num_classes=22)
                prediction_position_mask = torch.zeros((1, m_len))    # for the position, start with no substitution and add the human substitutions
                subst_positions = []
                tot_positions = 0    # counting positions without gap for the subst rate
                for i, v in enumerate(prediction_seq[0]):
                    if prediction_seq[0,i] != 21 and anchor_seq[0,i] != 21:
                        tot_positions += 1
                        if prediction_seq[0,i] != anchor_seq[0,i]:
                            subst_positions.append(i)
                if tot_positions > 0:
                    human_subst_rate = len(subst_positions) / tot_positions
                    if human_subst_rate > proxy_subst_rate:    # in this case, reduce the number of substitutions in the prediction
                        subst_nb = int(proxy_subst_rate*tot_positions)
                        subst_positions = random.choices(subst_positions, k=subst_nb)
                    for p in np.array(subst_positions):
                        prediction_position_mask[0,p] = 1
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

    def print_transitionMatrix(self):
        with open('transition_matrix.txt', 'w') as f:
            letters = residue_constants.aatype_to_str_sequence(list(range(22)))
            f.write(letters)
            for i in range(len(self.transitionMatrix)):
                for j in range(len(self.transitionMatrix[0])):
                    f.write(str(self.transitionMatrix[i,j].item()) + '\t')
                f.write('\n')
            f.write('\n')
            f.write(np.array2string(self.transitionMatrix.numpy()))
            f.write('\n')
            f.write('argmax: ')
            for i in range(20):
                f.write(residue_constants.aatype_to_str_sequence([i]))
                f.write(':')
                f.write(residue_constants.aatype_to_str_sequence([self.predict_aa([i])]))
                f.write('\t')
        
    def updateTransitionMatrix(self, target_seq: torch.Tensor, anchor_seq: torch.Tensor):
        for i in range(target_seq.size()[0]):
            if anchor_seq[i] != target_seq[i]:
                self.transitionMatrix[anchor_seq[i], target_seq[i]] += 1

    def predict_aa(self, anchor_aa):
        transition_noGaps = self.transitionMatrix[:20,:20]
        return torch.argmax(transition_noGaps[anchor_aa])


class TransitionRandomModule():
    def __init__(self):
        self.transition = {}
        codontab = my_functions.codontab
        aa_subst_count = {}
        for c in codontab.keys():
            aa = codontab[c]
            for subst in self.all_codon_subst(c):
                if codontab[subst] != aa and codontab[subst] != '*':
                    if aa not in aa_subst_count.keys():
                        aa_subst_count[aa] = [codontab[subst]]
                    else:
                        aa_subst_count[aa].append(codontab[subst])
        for aa in aa_subst_count.keys():
            all_subst = aa_subst_count[aa]
            print(aa, all_subst)
            best = max(set(all_subst), key = all_subst.count)
            self.transition[aa] = best
        print(self.transition)
            
    def all_codon_subst(self, c):
        subst = []
        for i, nuc in enumerate(c):
            for new_nuc in 'ACGT':
                if new_nuc != nuc:
                    new_c = c[:i]+new_nuc+c[i+1:]
                    subst.append(new_c)
        return subst
                    
    def predict_aa(self, aa):
        return self.transition[aa]
