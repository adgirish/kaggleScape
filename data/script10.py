
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

#from fastai libraries
# from fastai.imports import *
# from fastai.structured import *

import pandas as pd
# from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
# from treeinterpreter import treeinterpreter as ti
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from scipy.cluster import hierarchy as hc
import os
import numpy as np


# # Data Loading

# In[ ]:


types_dict_train = {'train_id': 'int64',
             'item_condition_id': 'int8',
             'price': 'float64',
             'shipping': 'int8'}


# In[ ]:


train = pd.read_csv('../input/train.tsv',delimiter='\t',low_memory=True,dtype=types_dict_train)


# In[ ]:


types_dict_test = {'test_id': 'int64',
             'item_condition_id': 'int8',
             'shipping': 'int8'}


# In[ ]:


test = pd.read_csv('../input/test.tsv',delimiter='\t',low_memory= True,dtype=types_dict_test)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape,test.shape #


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)


# In[ ]:


display_all(train.describe(include='all').transpose())


# In[ ]:


# train_cats(train)


# In[ ]:


# train_cats(test)


# In[ ]:


train.category_name = train.category_name.astype('category')
train.item_description = train.item_description.astype('category')

train.name = train.name.astype('category')
train.brand_name = train.brand_name.astype('category')


# In[ ]:


test.category_name = test.category_name.astype('category')
test.item_description = test.item_description.astype('category')

test.name = test.name.astype('category')
test.brand_name = test.brand_name.astype('category')


# In[ ]:


train.dtypes


# In[ ]:


test.dtypes


# In[ ]:


train.apply(lambda x: x.nunique())


# In[ ]:


test.apply(lambda x: x.nunique())


# In[ ]:


train.isnull().sum(),train.isnull().sum()/train.shape[0]


# In[ ]:


test.isnull().sum(),test.isnull().sum()/test.shape[0]


# In[ ]:


os.makedirs('data/tmp',exist_ok=True)


# In[ ]:


# train.to_feather('data/tmp/train_raw')

# test.to_feather('data/tmp/test_raw')


# # Model Building

# In[ ]:


# train = pd.read_feather('data/tmp/train_raw')
# test = pd.read_feather('data/tmp/test_raw')


# In[ ]:


train = train.rename(columns = {'train_id':'id'})


# In[ ]:


train.head()


# In[ ]:


test = test.rename(columns = {'test_id':'id'})


# In[ ]:


test.head()


# In[ ]:


train['is_train'] = 1
test['is_train'] = 0


# In[ ]:


train_test_combine = pd.concat([train.drop(['price'],axis =1),test],axis = 0)


# In[ ]:


train_test_combine.category_name = train_test_combine.category_name.astype('category')
train_test_combine.item_description = train_test_combine.item_description.astype('category')

train_test_combine.name = train_test_combine.name.astype('category')
train_test_combine.brand_name = train_test_combine.brand_name.astype('category')


# In[ ]:


train_test_combine = train_test_combine.drop(['item_description'],axis = 1)


# In[ ]:


train_test_combine.name = train_test_combine.name.cat.codes


# In[ ]:


train_test_combine.category_name = train_test_combine.category_name.cat.codes


# In[ ]:


train_test_combine.brand_name = train_test_combine.brand_name.cat.codes


# In[ ]:


# train_test_combine.item_description = train_test_combine.item_description.cat.codes


# In[ ]:


train_test_combine.head()


# In[ ]:


train_test_combine.dtypes


# In[ ]:


df_test = train_test_combine.loc[train_test_combine['is_train']==0]
df_train = train_test_combine.loc[train_test_combine['is_train']==1]


# In[ ]:


df_test = df_test.drop(['is_train'],axis=1)


# In[ ]:


df_train = df_train.drop(['is_train'],axis=1)


# In[ ]:


df_test.shape


# In[ ]:


df_train.shape


# In[ ]:


df_train['price'] = train.price


# In[ ]:


df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)


# In[ ]:


df_train.head()


# In[ ]:


# df_train.to_feather('data/tmp/train_raw_pro')


# In[ ]:


# df_test.to_feather('data/tmp/test_raw_pro')


# In[ ]:


# df_train = pd.read_feather('data/tmp/train_raw_pro')
# df_test = pd.read_feather('data/tmp/test_raw_pro')


# In[ ]:


x_train,y_train = df_train.drop(['price'],axis =1),df_train.price


# In[ ]:


# reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_jobs=-1,min_samples_leaf=3,n_estimators=200)
m.fit(x_train, y_train)
m.score(x_train,y_train)


# In[ ]:


preds = m.predict(df_test)


# In[ ]:


preds = pd.Series(np.exp(preds))


# In[ ]:


type(preds)


# In[ ]:


submit = pd.concat([df_test.id,preds],axis=1)


# In[ ]:


submit.columns = ['test_id','price']


# In[ ]:


submit.to_csv("./rf_v3.csv", index=False)

