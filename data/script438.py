
# coding: utf-8

# # Some initial features from the Variants
# * Note: As this is my first published Kernel, forgive my coding style and brevity.
# 
# The bulk of the features I've generated so far involve the mutations/Variants and their effects. My background is in Proteins/Neuropeptides, so I took some ideas from my Thesis, which was a number of Feature engineering libraries for Protein sequences. 
# 
# You can read up on feature engineering/ML with proteins/peptides in one of the papers, and the code+toolkits are all freely available (NeuroPID, ProFET, ASAP).  If you use it, please cite :). 
# 
# * NeuroPID: a predictor for identifying neuropeptide precursors
#     * https://www.ncbi.nlm.nih.gov/pubmed/24336809
# * ProFET: Feature engineering captures high-level protein functions.
# 
#         * https://www.ncbi.nlm.nih.gov/pubmed/26130574
# 
#         * https://github.com/ddofer/ProFET/blob/master/ProFET/feat_extract/AAlphabets.py
# 
# 
# I'll try to add some additional techniques/tricks as the competition progresses, but no promises. (I'm lazy, and my code is not just in Python. I'm hoping to lead by example here, and that others will upload rather better code than this).

# In[ ]:


import os
import math
import numpy as np
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv("../input/training_variants")
test = pd.read_csv("../input/test_variants")
#train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
#test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

print("Train shape".ljust(15), train.shape)
print("Test shape".ljust(15), test.shape)


# ### Joint train and test for our feature extraction

# In[ ]:


df_joint = pd.concat([train,test])
print("train+test rows:",df_joint.shape[0])

genes = set(df_joint["Gene"])
print("%i unique Genes" %(len(genes)))

variations = set(df_joint["Variation"])
print("%i unique Variations" %(len(variations)))


# # Variations FE:
# * extract amino acid letters and positions from variation. 
# * not all variations in this format. 
# * Some cases: the first letter may still give a signal; may be better to filter by the mutation not being in a list of meaningful words: "Truncating Mutations", "deletion', etc'.  (some are single some multi words - not trivial to seperate).
# * Future work: add physiochemical distance between affected AA using substitution matrices (requires munging local .txt matrix file and uploading, currently not ready for public kernel due to file dependency).
#     * Simple regex to capture our patterns:  
#  ^[A-Z]\d{1,7}[A-Z]
#         * ^ = start of string
#             \d{1,7}  = 1 to 7 digits
#             * df_joint.loc[df_joint["Variation"].str.contains(r'^[A-Z]\d{1,7}[A-Z]')]
# 
#     * https://stackoverflow.com/questions/6229906/regular-expression-to-match-integers-up-to-9-digits ->  
# 
#     * Maybe filter by string containing number? 
#     * Could filter in some cases by multiword ; len>5. 
# * Partial bad subwords list:
#     *  Fusion ,  Overexpression , Epigenetic Silencing, DNA, Amplification,  'Wildtype','Single Nucleotide Polymorphism','insertions/deletions , 'insertion', "duplications", "Mutations", "Truncating", "Deletion", "Hypermethylation" , "Copy Number Loss", 'MYC-nick'
#     
#     * standard 20 aa (not counting unknowns, or non standard/rare): AA_VALID = 'ACDEFGHIKLMNPQRSTVWY'

# In[ ]:


"Could use bioPython's IUPAC alphabet. This is simpler for now, although realistically we might want to handle non standard AA (e.g. selenocysteine, B,U,Z..)"
AA_VALID = 'ACDEFGHIKLMNPQRSTVWY'


# In[ ]:


df_joint["simple_variation_pattern"] = df_joint.Variation.str.contains(r'^[A-Z]\d{1,7}[A-Z]',case=False)


# In[ ]:


print("We capture most variants with this (and possibly even more in train)")
df_joint["simple_variation_pattern"].describe()


# In[ ]:


df_joint[df_joint["simple_variation_pattern"]==False]["Variation"].head(15)


# ##### Code notes:
# * Adding the logical condition for:  row.simple_variation_pattern==True didn't filter correctly for me, so I'm doing it the slow way.
# * It be better to capture the regex, then extract it (especially for the second/last letter clause).
# * there are many lot of edge cases we're not matching, or different versions. 
# 
# * Code could likely be improved in general with better regex, extraction get first word.. :
# https://stackoverflow.com/questions/37504672/pandas-dataframe-return-first-word-in-string-for-column
# 
# * Additional edge cases include: W802*	
# 
# 
# ### Length and relative location in gene: 
# * Proper usage would require external data on actual length of gene. 
# * ASAP has good code for such highly localized features, and resources.
#  *  "ASAP: A Machine-Learning Framework for Local Protein Properties" Dan Ofer, Nadav Brandes, Michal Linial doi: http://dx.doi.org/10.1101/032532
#  * https://github.com/ddofer/asap

# In[ ]:


# Get location in gene / first number , from first word (otherwise numbers appear later)
df_joint['location_number'] = df_joint.Variation.str.extract('(\d+)')


# In[ ]:


df_joint['variant_letter_first'] = df_joint.apply(lambda row: row.Variation[0] if row.Variation[0] in (AA_VALID) else np.NaN,axis=1)
df_joint['variant_letter_last'] = df_joint.apply(lambda row: row.Variation.split()[0][-1] if (row.Variation.split()[0][-1] in (AA_VALID)) else np.NaN ,axis=1)


# In[ ]:


df_joint['variant_letter_last'].describe()


# In[ ]:


df_joint[['variant_letter_first',"Variation",'variant_letter_last',"simple_variation_pattern"]].head(4)


# In[ ]:


" Replace letters with NaNs for cases that don't match our pattern. (Need to check if this actually improves results!)"
df_joint.loc[df_joint.simple_variation_pattern==False,['variant_letter_last',"variant_letter_first"]] = np.NaN


# ## Reduced AA alphabet distance FE
# * Check if the mutated AA are in the same reduced amino acid alphabet groups, from NeuroPID or ProFET
#     * http://www.protonet.cs.huji.ac.il/neuropid/code/index.php
#     * NeuroPID: a predictor for identifying neuropeptide precursors
#         * https://www.ncbi.nlm.nih.gov/pubmed/24336809
#     * ProFET: Feature engineering captures high-level protein functions.
#         * https://www.ncbi.nlm.nih.gov/pubmed/26130574
#         * https://github.com/ddofer/ProFET/blob/master/ProFET/feat_extract/AAlphabets.py
# * Can also check for AA distance. 
#     * https://gist.github.com/arq5x/5408712
# 

# In[ ]:


"""
## Bioinformatics Code + alphabet feature engineering from: https://github.com/ddofer/ProFET/blob/master/ProFET/feat_extract/AAlphabets.py

ProFET: Feature engineering captures high-level protein functions.
Ofer D, Linial M.
Bioinformatics. 2015 Nov 1;31(21):3429-36. doi: 10.1093/bioinformatics/btv345.
PMID: 26130574
"""

def TransDict_from_list(groups):
    '''
    Given a list of letter groups, returns a dict mapping each group to a
    single letter from the group - for use in translation.
    >>> alex6=["C", "G", "P", "FYW", "AVILM", "STNQRHKDE"]
    >>> trans_a6 = TransDict_from_list(alex6)
    >>> print(trans_a6)
    {'V': 'A', 'W': 'F', 'T': 'D', 'R': 'D', 'S': 'D', 'P': 'P',
     'Q': 'D', 'Y': 'F', 'F': 'F',
     'G': 'G', 'D': 'D', 'E': 'D', 'C': 'C', 'A': 'A',
      'N': 'D', 'L': 'A', 'M': 'A', 'K': 'D', 'H': 'D', 'I': 'A'}
    '''
    transDict = dict()

    result = {}
    for group in groups:
        g_members = sorted(group) #Alphabetically sorted list
        for c in g_members:
            result[c] = str(g_members[0]) #K:V map, use group's first letter as represent.
    return result

ofer8=TransDict_from_list(["C", "G", "P", "FYW", "AVILM", "RKH", "DE", "STNQ"])

sdm12 =TransDict_from_list(
    ["A", "D", "KER", "N",  "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"] )

pc5 = {"I": "A", # Aliphatic
         "V": "A",         "L": "A",
         "F": "R", # Aromatic
         "Y": "R",         "W": "R",         "H": "R",
         "K": "C", # Charged
         "R": "C",         "D": "C",         "E": "C",
         "G": "T", # Tiny
         "A": "T",         "C": "T",         "S": "T",
         "T": "D", # Diverse
         "M": "D",         "Q": "D",         "N": "D",
         "P": "D"}


# In[ ]:


"You can encode the reduced alphabet as OHE features; in peptidomics this gives highly generizable features."
df_joint['AAGroup_ofer8_letter_first'] = df_joint["variant_letter_first"].map(ofer8)
df_joint['AAGroup_ofer8_letter_last'] = df_joint["variant_letter_last"].map(ofer8)
df_joint['AAGroup_ofer8_equiv'] = df_joint['AAGroup_ofer8_letter_first'] == df_joint['AAGroup_ofer8_letter_last']

df_joint['AAGroup_m12_equiv'] = df_joint['variant_letter_last'].map(sdm12) == df_joint['variant_letter_first'].map(sdm12)
df_joint['AAGroup_p5_equiv'] = df_joint['variant_letter_last'].map(pc5) == df_joint['variant_letter_first'].map(pc5)


# In[ ]:


df_joint['AAGroup_ofer8_equiv'].describe()


# Unsurprisingly, most of the mutations are not considered functionally equivalent. 
# 
# * This makes sense, as highly similar mutations are not good candidates for the sort of studies and manual annotation we'd expect from the challenge's authors.
# 
# * Warning! The equivalence currently doesn't handle (*) or NaNs, so it'll consider NaN=NaN to be true! 
# 
# 

# ### Amino Acid Evolutionary Distance matrix features:
# * https://gist.github.com/arq5x/5408712
# * Requires files of the distance matrix + py 2.7 cleanup; not my code. 

# In[ ]:


print(df_joint.shape)
df_joint.head()


# In[ ]:


train = df_joint.loc[~df_joint.Class.isnull()]
test = df_joint.loc[df_joint.Class.isnull()]


# In[ ]:


train.to_csv('train_variants_featurized_raw.csv', index=False)
test.to_csv('test_variants_featurized_raw.csv', index=False)

