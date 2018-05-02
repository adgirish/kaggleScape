
# coding: utf-8

# # Imbalanced binary data - Let's Downsample the majority class!
# * The data is extremely imbalanced (0.17% positives).
# * Data is also quite large, and anonymized (so we can't get many meaningful aggregations from the anonymized variables).
# * We'll downsample/negative sample by class.
# * AUC doesn't care about class balance, only seperation!
# * This will let us work with a much smaller, faster dataset for nice fast iterations :) 
# 
#     * NOTE! This sort of approach won't always work. However, when the minority class is far more interesting, and there's this much data, and it's so imbalanced, etc', then this sort of downsampling is an excellent appraoch.
#         * If we cared about logloss (or had some other evaluation metric, such as TPR/FPR etc'), then we could alwyas weight our data/sample/ correct the evaluation

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask
import os
print(os.listdir("../input"))


# pd.merge(pd.read_csv('../input/test.csv', dtype=dtypes),
# pd.read_csv('../input/train.csv', dtype=dtypes).groupby(['app','channel'])['is_attributed'].mean().reset_index(),
# on=['app','channel'], how='left').fillna(0)[['click_id','is_attributed']].to_csv('submean_app.csv', index=False)
# # Any results you write to the current directory are saved as output.


# In[ ]:


dtypes = {
        'ip':'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8'
        }


# ## Read in train data
# * We  read it all at once then split into 2 dataframes, rather than iterating over the file twice. Note that this will mean an extra copy of the data in our RAM. 
# * Dask (or maybe ray?) could speed this up nicely:
#     * https://stackoverflow.com/questions/34173859/filtering-a-large-dataframe-in-pandas-using-multiprocessing 

# In[ ]:


# df_train = pd.read_csv('../input/train.csv', nrows=10000)
# df_test = pd.read_csv('../input/test.csv', nrows=10000)
# df_train.head()
# df_test.head()


# ##### Helper function to get mean values, and apply them also on test
# * We could do this more elgantly using pandas's .transform("mean") , but then we'd need to merge it onto the test, and that's more annoying. 
# * Note that this can easily overfit! (Better to use smoothing, e.g. bayesian average).
# * Note also that we can get feature crosses with this. (e.g. "app=iphone AND ip=192.168")

# In[ ]:


# https://www.kaggle.com/cttsai/blend-app-channel-and-app-mean

def mean_feat(train, test, attrs=[]):
    return pd.merge(test, train.groupby(attrs)['is_attributed'].mean().reset_index(), on=attrs, how='left').fillna(0).set_index('click_id')


# In[ ]:


import dask.dataframe as dd
dask_df = dd.read_csv('../input/train.csv',dtype=dtypes)
dask_df.npartitions


# In[ ]:


df_pos = dask_df[(dask_df['is_attributed'] == 1)].compute()
print("Total positives : ",df_pos.shape[0])
df_neg = dask_df[(dask_df['is_attributed'] == 0)].compute()
print("Total Negatives : ",df_neg.shape[0])
print("Base percentage of positives : ",100*df_pos.shape[0]/df_neg.shape[0])


# In[ ]:


df_neg = df_neg.sample(n=3000000) # 2.25 million = 20% , 4.5 = ~10%


# In[ ]:


# join downsampled data and shuffle them
df = pd.concat([df_pos,df_neg]).sample(frac=1)
print(df.shape)


# In[ ]:


df.to_csv("train_downsampled_3m.csv.gz",index=False,compression="gzip")


# # As additional, prior steps we could calculate out aggregate features BEFORE the downsampling.
# * Likely to be usefu only for long tail, such as pairwise features.

# In[ ]:


# # df = pd.concat([df_pos.sample(n=50000),df_neg.sample(n=100000)])
# df = pd.concat([df_pos.head(n=50000),df_neg.head(n=100000)])
# df.head()


# In[ ]:


## code gives error? 

def group_mean(df,categoricals, target):

	for col in categoricals:

		grouped = df[target,col].groupby([col])
		n = grouped[target].transform('count')
		mean = grouped[target].transform('mean')
		df['%s_result'%(col)] = (mean*n - df[target])/(n-1)
	return df
# group_mean(df,categoricals=["app","os"], target="is_attributed")


# In[ ]:


# group_mean(df,categoricals=["app","os"], target="is_attributed")

