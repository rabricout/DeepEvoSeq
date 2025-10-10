def aa_attributes(aa):
    petit = ['P', 'N', 'D', 'S', 'C', 'T', 'A', 'G', 'V']
    minuscule = ['A', 'G', 'C', 'S']
    proline = ['P']
    hydrophobe = ['M', 'K', 'F', 'Y', 'W', 'H', 'I', 'L', 'V', 'T', 'C', 'G', 'A']
    polaire = ['Y', 'W', 'H', 'K', 'R', 'D', 'E', 'T', 'C', 'S', 'N', 'Q']
    charges = ['D', 'E', 'H', 'K', 'R']
    negatifs = ['D', 'E']
    positifs = ['H', 'K', 'R']
    aromatiques = ['F', 'Y', 'W', 'H']
    aliphatiques = ['I', 'L', 'V']
    
    attributes = []
    if aa in petit:
        attributes.append('small')
    if aa in minuscule:
        attributes.append('tiny')
    if aa in proline:
        attributes.append('proline')
    if aa in hydrophobe:
        attributes.append('hydrophobic')
    if aa in polaire:
        attributes.append('polar')
    if aa in charges:
        attributes.append('charged')
    if aa in negatifs:
        attributes.append('negative')
    if aa in positifs:
        attributes.append('positive')
    if aa in aromatiques:
        attributes.append('aromatic')
    if aa in aliphatiques:
        attributes.append('aliphatic')

    return attributes

#def compare_codons_attributes(c1, c2):
#    return compare_aa_attributes(codon_to_aa(c1), codon_to_aa(c2))

#def compare_aa_attributes(aa1, aa2):
#    aa1_att = aa_attributes(aa1)
#    aa2_att = aa_attributes(aa2)
#    diff = []
#    for att in aa1_att:
#        if att not in aa2_att:
#            diff.append(att)
#    for att in aa2_att:
#        if att not in aa1_att:
#            diff.append(att)
#    return diff
