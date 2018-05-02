
# coding: utf-8

# Seems like the best way to get a good LB score is to blend a bunch of good predictions with low correlations. I noticed that a few kernals with high LB scores (0.9865+) are carefully weighing individual predictions to achieve their results. I wanted to try something similar but I was feeling lazy. Examining all the correlations for all the classes for my 20+ predictions seemed tedious. And I was worried that manually choosing weights based on LB scores would result in overfitting to the public LB. So I came up with a neat, automated way to ensemble any number of predictions without having to look at them.
# 
# **General idea:**
# 1. For every class (toxic,severe_toxic,obscene,etc.):
#     2. Compute the correlation between every pair of files (predictions)
#     3. Merge the two files with the highest correlations into a new file (using the average of their predictions)
#     4. Delete the two files that were merged (keep the file representing their average)
#     6. Repeat steps 3-5 until only one file is left
# 
# The result is a blend of all 20+ predictions with the least correlated files recieving the most "weight."
# 
# **Refinements: **
# * The least correlated file could be weighted as high as 50% which seems a little too extreme. So I started tracking the "density" of each file which represents the number of files that went into it. I added a `DENSITY_COEFF` parameter that controls how much density is considered when merging two files. For example:
#     * `DENSITY_COEFF = 1.0`:  density of a file not considered at all. One file *could* receive 50% of the weight.
#     * `DENSITY_COEFF = 0.0`:  correlation (merge order) isn't considered at all. Every file receives equal weight. Just an average of all predictions.
# * Very similar predictions are merged first and build up a large density wich can overrule later, less correlated predictions. To fix this I added a cutoff threshold: `OVER_CORR_CUTOFF`. When the correlation between two files is above the threshold the weight of the file resulting from their merger is the maximum weight of the two instead of the sum. (This also solves the problem of accidentially including the same file twice. Hooray laziness!)
# 
# **Code:**
# 

# In[ ]:


import pandas as pd
import numpy as np
import os

# Controls weights when combining predictions
# 0: equal average of all inputs; 
# 1: up to 50% of weight going to least correlated input
DENSITY_COEFF = 0.1
assert DENSITY_COEFF >= 0.0 and DENSITY_COEFF <= 1.0

# When merging 2 files with corr > OVER_CORR_CUTOFF 
# the result's density is the max instead of the sum of the merged files' densities
OVER_CORR_CUTOFF = 0.98
assert OVER_CORR_CUTOFF >= 0.0 and OVER_CORR_CUTOFF <= 1.0

INPUT_DIR = '../input/private-toxic-comment-sumbmissions/'

def load_submissions():
    files = os.listdir(INPUT_DIR)
    csv_files = []
    for f in files:
        if f.endswith(".csv"):
            csv_files.append(f)
    frames = {f:pd.read_csv(INPUT_DIR+f).sort_values('id') for f in csv_files}
    return frames


def get_corr_mat(col,frames):
    c = pd.DataFrame()
    for name,df in frames.items():
        c[name] = df[col]
    cor = c.corr()
    for name in cor.columns:
        cor.set_value(name,name,0.0)
    return cor


def highest_corr(mat):
    n_cor = np.array(mat.values)
    corr = np.max(n_cor)
    idx = np.unravel_index(np.argmax(n_cor, axis=None), n_cor.shape)
    f1 = mat.columns[idx[0]]
    f2 = mat.columns[idx[1]]
    return corr,f1,f2


def get_merge_weights(m1,m2,densities):
    d1 = densities[m1]
    d2 = densities[m2]
    d_tot = d1 + d2
    weights1 = 0.5*DENSITY_COEFF + (d1/d_tot)*(1-DENSITY_COEFF)
    weights2 = 0.5*DENSITY_COEFF + (d2/d_tot)*(1-DENSITY_COEFF)
    return weights1, weights2


def ensemble_col(col,frames,densities):
    if len(frames) == 1:
        _, fr = frames.popitem()
        return fr[col]

    mat = get_corr_mat(col,frames)
    corr,merge1,merge2 = highest_corr(mat)
    new_col_name = merge1 + '_' + merge2

    w1,w2 = get_merge_weights(merge1,merge2,densities)
    new_df = pd.DataFrame()
    new_df[col] = (frames[merge1][col]*w1) + (frames[merge2][col]*w2)
    del frames[merge1]
    del frames[merge2]
    frames[new_col_name] = new_df

    if corr >= OVER_CORR_CUTOFF:
        print('\t',merge1,merge2,'  (OVER CORR)')
        densities[new_col_name] = max(densities[merge1],densities[merge2])
    else:
        print('\t',merge1,merge2)
        densities[new_col_name] = densities[merge1] + densities[merge2]

    del densities[merge1]
    del densities[merge2]
    #print(densities)
    return ensemble_col(col,frames,densities)


ens_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv').sort_values('id')
#print(get_corr_mat('toxic',load_submissions()))

for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]:
    frames = load_submissions()
    print('\n\n',col)
    densities = {k:1.0 for k in frames.keys()}
    ens_submission[col] = ensemble_col(col,frames,densities)

print(ens_submission)    
ens_submission.to_csv('lazy_ensemble_submission.csv', index=False)


# I've just been dumping all my good prediction files into a folder and then re-running the code above. It seems to do pretty well. I figure with only 2 parameters to tweak (`DENSITY_COEFF` and `OVER_CORR_CUTOFF`) it's less likely to overfit to the public LB compared to selecting and weighting specific files.
# 
# 
# I'm fairly new to this Data Science stuff so I'd love to get feedback on this! 
# Specifically:
# * Does this sort of approch make sense?
# * How can it be improved?
# * Am I reinventing the wheel? Is there already a library/algorithm that does this? 
# 
# 
# **Acknowledgements:**
# 
# Huge thanks to everyone published a unique model! There's a good chance I'm including your work in my ensemble. 
