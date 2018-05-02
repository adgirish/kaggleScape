
# coding: utf-8

# ## Temporal corelations
# 
# The idea of this notebook is to see how corelation between various features and target variable change over time.
# 
# As has been observed in other EDA notebooks, there is very little corelation between given features and y variable.
# 
# As pointed by Raddar here, distribution of features seem to vary quite a bit for many features with time.

# In[ ]:


import kagglegym
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib import rcParams
rcParams['figure.figsize'] = 8, 6


# In[ ]:


train_all = pd.read_hdf('../input/train.h5')

#Skipping alternate timestamps to be able to run this kernel

#odd_timestamps = [t for t in  train_all.timestamp.unique() if t%2 !=0]
#print(len(odd_timestamps))

#train_all = train_all.loc[train_all.timestamp.isin(odd_timestamps)]


# In[ ]:


#Lets get all features and devide into derived, fundamental and technical categories
feats = [col for col in train_all.columns if col not in ['id', 'timestamp', 'y']]


# It would be good idea to do a box plot of all variables after clip them appropriately (as there are lot of very high values).

# In[ ]:


train_all[feats] = train_all[feats].clip(upper=5, lower=-5)


# In[ ]:


train_all[feats].plot(kind='box')
plt.ylim([-2,2])


# All the features lie mostly within -2 to 2. So, clipping features at -2 and 2 could be a good idea for modeling purposes

# In[ ]:


#Group by mean, median and 2 sigma. (Note! Many of the features are not normal, 
#hence 2 sigma  does not give complete picture but good for starters) 
#(No pun intended for sponsor of competition ;)

train_all_mean = train_all.groupby('timestamp').apply(lambda x: x.mean())

train_all_2stdp = train_all.groupby('timestamp').apply(lambda x: x.mean() + 2*x.std())

train_all_2stdm = train_all.groupby('timestamp').apply(lambda x: x.mean() - 2*x.std())


# In[ ]:


tmp1 = pd.melt(train_all_mean, value_vars=feats, var_name='features1', value_name='mean')
tmp2 = pd.melt(train_all_2stdp, value_vars=feats, var_name='features2', value_name='2_sigma_plus')
tmp3 = pd.melt(train_all_2stdm, value_vars=feats, var_name='features3', value_name='2_sigma_minus')

tmp = pd.concat([tmp1, tmp2, tmp3], axis=1)

fg = sns.FacetGrid(data=tmp, col='features1', col_wrap=3, size=2.7)
fg.map(plt.plot, 'mean', color='blue')
fg.map(plt.plot, '2_sigma_plus', color='black')
fg.map(plt.plot, '2_sigma_minus', color='black')
#fg.map(plt.fill_between, 'mean', '2_sigma_minus', '2_sigma_plus', color='Purple', alpha=0.3)

#plt.ylim([-4.5, 4.5])

del tmp1, tmp2, tmp


# * As has been already been pointed out by raddar, variables that are stable (flat mean and variance) over time are more suitable for modeling.**
# 
# * Removing(or modeling separately) two periods of high variance in y, can provide additional paramters for modeling 
# 
# 
# Now, lets see how corelation with y changes for different variables

# In[ ]:


#I had to impute missing data otherwise either it takes very long time or we get lot of NaN 
import time
def get_corr(x):
    s= time.time()
    #for f in feats:
    #    corr.append(np.corrcoef(x[f],x['y'],rowvar=0)[0,1])
    corr = np.corrcoef(x.values.T)[-1,2:-1]
    #print(time.time()- s)
    return corr
    
train_all_imputed = train_all.fillna(0)
train_all_corr = train_all_imputed.groupby('timestamp').apply(get_corr) #This will take some time
train_all_corr = pd.DataFrame(np.vstack(train_all_corr), columns=feats)
train_all_corr.head()


# In[ ]:


tmp3 = pd.melt(train_all_corr, value_vars=feats, var_name='features3', value_name='corr')

fg = sns.FacetGrid(data=tmp3, col='features3', col_wrap=3, size=2.8)
fg.map(plt.plot, 'corr', color='blue').add_legend()
del tmp3


# All the corelation coefficients are oscillating quite a lot. Almost all features show positive and negative corelations with high frequency, most likely because of high volatility of y.

# In[ ]:


def get_corr2(x):
    corr = np.corrcoef(x.values.T)[-1,2:-2]
    return corr
    
    
train_all_imputed['abs_y'] = abs(train_all_imputed['y'])
train_all_corr2 = train_all_imputed.groupby('timestamp').apply(get_corr2) #This will take some time
train_all_corr2 = pd.DataFrame(np.vstack(train_all_corr2), columns=feats)
train_all_corr2.head()


# In[ ]:


tmp4 = pd.melt(train_all_corr2, value_vars=feats, var_name='features3', value_name='corr')

fg = sns.FacetGrid(data=tmp4, col='features3', col_wrap=3, size=2.7)
fg.map(plt.plot, 'corr', color='blue').add_legend()

del tmp4


# ** Many features are showing strong corelations with absolute values of y. **
# 
# ** Building separate models for absolute values of y and direction of y can be a good approach **
# 
# Also, they seem to be varying over time.**
# 
# Normalize everything and plot together

# In[ ]:


tmp1 = pd.melt(train_all_mean, value_vars=feats, var_name='features1', value_name='mean')
tmp2 = pd.melt(train_all_2stdp, value_vars=feats, var_name='features2', value_name='2std')
tmp3 = pd.melt(train_all_corr, value_vars=feats, var_name='features3', value_name='corr')
tmp4 = pd.melt(train_all_corr2, value_vars=feats, var_name='features3', value_name='yabs_corr')

tmp = pd.concat([tmp1, tmp2, tmp3, tmp4], axis=1)
cols = ['mean', '2std', 'corr', 'yabs_corr']
tmp[cols] = tmp[cols].apply(lambda x: (x - x.mean())/x.std())
fg = sns.FacetGrid(data=tmp, col='features1', col_wrap=3, size=2.7)
fg = fg.map(plt.plot, 'mean', color='red')
fg = fg.map(plt.plot, 'corr', color='green', alpha=0.4)
fg = fg.map(plt.plot, '2std', color='black')
fg = fg.map(plt.plot, 'yabs_corr', color='purple', alpha=0.4)
fg.add_legend()


del tmp1, tmp2, tmp3, tmp4, tmp


# **Many features have same trend as their correlation with absolute values of y. Some sort of temporal correction for features can improve lineal models **
