
# coding: utf-8

# Since this solution uses a Naive Bayesian Network to predict the target I select the features based on their mutual information with the target and their relative indepedence from each other.  This will strengthen the biggest assumption of any Naive BN model: all features are independent. But in this data there is wide dependency among the features. 
# 
# My approach to finding 'relatively' independent features is to first group the features into Concepts. My exploration has shown about 14 Concepts in the data (ps: I excluded all _calc_ features as white noise). To identify Concepts I first learn a BN model of the data with great restrictions on the learning process: features/nodes may have a single parent. Here is a DAG of such a model:
# 
# ![BN 4 showing concepts](http://elmtreegarden.com/wp-content/uploads/2017/03/bn4-w-concepts.jpg)
# 
# The 14 Concepts are color coded. The small "?" symbol means the feature has missing data. The color coding is imperfect and is is difficult to see that ps_ind_16_bin and ps_ind_14 are different Concepts.
# 
# The mutual information of each feature with the target is calculated and the features with at least 0.001 bits of mutual information are selected from each Concept. A total of 11 features were selected and shown below. Sons and Spouses learning resulted in the final 7 feature model.
# 
# ![Final model](http://elmtreegarden.com/wp-content/uploads/2017/03/BN-5-w-all-fv-NB-22-8-gini-1.jpg)
# 
# Note the very large Entropy of car_11_cat ! This deserves much more exploration.

# In[ ]:


import pandas as pd
import numpy as np

#mdf = 'c:/Users/John/Documents/Kaggle/Porto Seguro/'
mdf = '../input/'

train = pd.read_csv(mdf + "train.csv", usecols = ['target', 'ps_car_07_cat',
    'ps_car_02_cat', 'ps_car_13','ps_reg_02', 'ps_ind_06_bin', 'ps_ind_16_bin', 'ps_ind_17_bin'])

test = pd.read_csv(mdf + "test.csv", usecols = ['id', 'ps_car_07_cat', 'ps_car_02_cat',
    'ps_car_13','ps_reg_02', 'ps_ind_06_bin', 'ps_ind_16_bin', 'ps_ind_17_bin'])
bins = [0.0, 0.639, 0.784, 1.093, 4.4]
train['ps_car_13_d'] = pd.cut(train['ps_car_13'], bins)
test['ps_car_13_d'] = pd.cut(test['ps_car_13'], bins)
bins2 = [-0.1, 0.25, 0.75, 2.0]
train['ps_reg_02_d'] = pd.cut(train['ps_reg_02'], bins2)
test['ps_reg_02_d'] = pd.cut(test['ps_reg_02'], bins2)
train.head(4)


# A Naive model was chosen in part because it is so easy to calculate in Python (any language really). It is a simple calculation of the conditional probability of the target given features 1 through 7:
# 
# p(target | feature1, ... , feature7) = p(target=1) x { p(feature1 | target) x ... x  p(feature7 | target) }  / Z
# 
# Z is a normalizing constant. It is calculated and used to normalize the whole prediction so that it has an average of about 3.75% (the average probability that target = 1).
# 

# In[ ]:


# Now calculate the each factor associated with each feature, (fi): p( feature_i | target)

f1 = pd.DataFrame()
f2 = pd.DataFrame()
f3 = pd.DataFrame()
f4 = pd.DataFrame()
f5 = pd.DataFrame()
f6 = pd.DataFrame()
f7 = pd.DataFrame()

f1 = train.groupby('ps_car_13_d')['target'].agg([('p_f1','mean')]).reset_index()
f2 = train.groupby('ps_reg_02_d')['target'].agg([('p_f2','mean')]).reset_index()
f3 = train.groupby(['ps_car_07_cat'])['target'].agg([('p_f3','mean')]).reset_index()
f4 = train.groupby(['ps_car_02_cat'])['target'].agg([('p_f4','mean')]).reset_index()
f5 = train.groupby('ps_ind_06_bin')['target'].agg([('p_f5','mean')]).reset_index()
f6 = train.groupby('ps_ind_16_bin')['target'].agg([('p_f6','mean')]).reset_index()
f7 = train.groupby('ps_ind_17_bin')['target'].agg([('p_f7','mean')]).reset_index()
f3.head(10)


# Above you can see that ps_car_07_cat has some missing data (notice the -1 state). My theory is that there is no missing data. Porto Seguro has given us all the information available. Therefore missing data is really a filtered state. It also seems from the DAG of Concepts that missing data seems concentrated in certain Concepts. This makes perfect sense if the missing data represents a filtered state.
# 
# 

# In[ ]:


sol1 = pd.DataFrame()
sol1 = test.merge(f1, on = 'ps_car_13_d')
sol2 = pd.DataFrame()
sol2 = sol1.merge(f2, on = 'ps_reg_02_d')
del sol1
sol3 = pd.DataFrame()
sol3 = sol2.merge(f3, on = 'ps_car_07_cat')
del sol2
sol4 = pd.DataFrame()
sol4 = sol3.merge(f4, on = 'ps_car_02_cat')
del sol3
sol5 = pd.DataFrame()
sol5 = sol4.merge(f5, on = 'ps_ind_06_bin')
del sol4
sol6 = pd.DataFrame()
sol6 = sol5.merge(f6, on = 'ps_ind_16_bin')
del sol5
sol = pd.DataFrame()
sol = sol6.merge(f7, on = 'ps_ind_17_bin')
del sol6
sol.head(5)


# In[ ]:


# f is the product of factors of feaures
sol.loc[:,'f'] = sol.loc[:,'p_f1'] * sol.loc[:,'p_f2'] * sol.loc[:,'p_f3'] * sol.loc[:,'p_f4']                 * sol.loc[:,'p_f5'] * sol.loc[:,'p_f6'] * sol.loc[:,'p_f7'] 

z = sol.f.sum() / len(sol.f)
# z is the normalizing factor
sol['target'] = 0.03645 * sol.loc[:,'f'] / z
sol[['id', 'target']].to_csv('bn_5_output_7_nodes.csv', index = False, float_format='%.4f')
sol.shape


# In[ ]:


# thanks to cpmpml for : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
from numba import jit

@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

sol1 = pd.DataFrame()
sol1 = train.merge(f1, on = 'ps_car_13_d')
sol2 = pd.DataFrame()
sol2 = sol1.merge(f2, on = 'ps_reg_02_d')
del sol1
sol3 = pd.DataFrame()
sol3 = sol2.merge(f3, on = 'ps_car_07_cat')
del sol2
sol4 = pd.DataFrame()
sol4 = sol3.merge(f4, on = 'ps_car_02_cat')
del sol3
sol5 = pd.DataFrame()
sol5 = sol4.merge(f5, on = 'ps_ind_06_bin')
del sol4
sol6 = pd.DataFrame()
sol6 = sol5.merge(f6, on = 'ps_ind_16_bin')
del sol5
sol = pd.DataFrame()
sol = sol6.merge(f7, on = 'ps_ind_17_bin')
del sol6
sol.loc[:,'f'] = sol.loc[:,'p_f1'] * sol.loc[:,'p_f2'] * sol.loc[:,'p_f3'] * sol.loc[:,'p_f4']                 * sol.loc[:,'p_f5'] * sol.loc[:,'p_f6'] * sol.loc[:,'p_f7'] 
z = sol.f.sum() / len(sol.f)
sol['exp_target'] = 0.03645 * sol.loc[:,'f'] / z

# Calculate GINI score
eval_gini(sol['target'], sol['exp_target'])

