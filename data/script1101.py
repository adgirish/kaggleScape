
# coding: utf-8

# I have commented a few parts of code which take more time e.g. gridsearchcv. I have also used fastai library to limit number of samples for each tree. (set_rf_samples  and reset_rf_samples -> function). Refer https://github.com/fastai for details. 
# 
# I have also taken basic preprocessing codes from the following kernel. 
# https://www.kaggle.com/shikhar1/base-random-forest-lb-532

# In[ ]:


def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))


# In[ ]:


def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))


# In[ ]:


# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


# This is required to make fastai library work on ec2-user 
# fastai library is not yet available using pip install
# pull it from github using below link
# https://github.com/fastai/courses
import warnings
warnings.filterwarnings('ignore')

#import sys
#sys.path.append("/Users/groverprince/Documents/msan/msan_ml/fastai/")
#sys.path.append("/home/groverprince/flogistix/fastai/")


# In[ ]:


# This file contains all the main external libs we'll use

import numpy as np
import pandas as pd

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
import operator


# In[ ]:


df_dtype = {}
df_dtype['train_id'] = 'int32'
df_dtype['item_condition_id'] = 'int32'
df_dtype['shipping'] = 'int8'
df_dtype['price'] = 'float32'
df_dtype


# In[ ]:


train = pd.read_csv("../input/train.tsv", delimiter='\t',
                   dtype= df_dtype)


# In[ ]:


train[:5]


# In[ ]:


train.apply(lambda x: x.nunique())


# In[ ]:


train.isnull().sum()


# In[ ]:


test = pd.read_csv("../input/test.tsv", delimiter='\t',
                   dtype= df_dtype)


# In[ ]:


test[:5]


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


len(test), len(train)


# In[ ]:


cat_vars = ['category_name', 'brand_name', 'shipping', 'name', 'item_description',
           'item_condition_id']


# In[ ]:


n = len(train)
n


# In[ ]:


train['is_train'] = 1
test['is_train'] = 0


# In[ ]:


train.rename(columns={'train_id':'id'}, inplace=True)
test.rename(columns={'test_id':'id'}, inplace=True)


# In[ ]:


train_test_combine = pd.concat([train.drop(['price'],axis =1),test],axis = 0)


# In[ ]:


for v in cat_vars: train_test_combine[v] = train_test_combine[v].astype('category').cat.as_ordered()


# In[ ]:


for v in cat_vars: train_test_combine[v] = train_test_combine[v].cat.codes


# In[ ]:


train_test_combine[:4]


# In[ ]:


df_test = train_test_combine.loc[train_test_combine['is_train']==0]
df_train = train_test_combine.loc[train_test_combine['is_train']==1]


# In[ ]:


df_test[:4]


# In[ ]:


df_train[:4]


# In[ ]:


df_train['price'] = train.price


# In[ ]:


df_train['price'] = df_train['price'].apply(lambda x: np.log(x) if x>0 else x)


# In[ ]:


df_train.drop(['id', 'is_train'],axis=1, inplace=True)


# In[ ]:


df_train[:3]


# In[ ]:


#df, y, nas, mapper = proc_df(df_train, 'price', do_scale=True)


# In[ ]:


#df[:4]


# In[ ]:


#y[:4]


# ## RF

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[ ]:


x_train,y_train = df_train.drop(['price'],axis =1),df_train.price


# In[ ]:


#set_rf_samples(50000)


# ### finding best n_estimators

# In[ ]:


#m = RandomForestRegressor(n_jobs=-1, n_estimators=125)
#%time m.fit(x_train, y_train)


# CPU times: user 1min 4s, sys: 1.95 s, total: 1min 6s
# Wall time: 21.6 s  
# Out[62]:  
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#            max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=125, n_jobs=-1,
#            oob_score=False, random_state=None, verbose=0, warm_start=False)

# In[ ]:


#preds = np.stack([t.predict(x_train) for t in m.estimators_])

#plt.plot([metrics.r2_score(y_train, np.mean(preds[:i+1], axis=0)) for i in range(125)]);
#plt.show()


# Based on above plot, I can select 30 trees for n_estimators in order to tune my rf parameters

# In[ ]:


grid = {
    'min_samples_leaf': [3,5,10,15,25,50,100],
    'max_features': ['sqrt', 'log2', 0.4, 0.5, 0.6]}


# In[ ]:


rf = RandomForestRegressor(n_jobs=-1, n_estimators=30,  random_state=42)


# In[ ]:


#gd = GridSearchCV(rf,grid, cv=3, verbose=50)


# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#            max_features=0.5, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=3, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=-1,
#            oob_score=False, random_state=42, verbose=0, warm_start=False)

# In[ ]:


#gd.fit(x_train, y_train)


# In[ ]:


#gd.best_estimator_


# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
#            max_features=0.5, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=3, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=-1,
#            oob_score=False, random_state=42, verbose=0, warm_start=False)

# **Max feature = 0.5 and min_samples_leaf = 3**

# In[ ]:


#reset_rf_samples() 
rf2 = RandomForestRegressor(n_jobs=-1, n_estimators=100,  random_state=42, max_features=0.5, min_samples_leaf=3)


# In[ ]:


rf2.fit(x_train,y_train)


# In[ ]:


rf2.score(x_train,y_train)


# 0.7552096320414714

# In[ ]:


df_test.drop(['is_train', 'id'], inplace=True, axis=1)


# In[ ]:


preds = rf2.predict(df_test)
preds = pd.Series(np.exp(preds))
submit = pd.concat([test.id,preds],axis=1)
submit.columns = ['test_id','price']


# In[ ]:


submit.to_csv('submit_rf_1.csv',index=False)


# In[ ]:


FileLink('submit_rf_1.csv')

