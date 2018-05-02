
# coding: utf-8

# # Computing Feature Imporatance From Scratch
# 
# Developing an initial machine learning algorithm that "works" can be easy, but developing a *good* machine learning algorithm may be difficult. One very helpful strategy is learning how to quantify the importance of a feature. By learning which features are most helpful in making decisions, we can improve our model or simply gain insight into our data. In this notebook, we will learn how to build feature importance from scratch using Random Forest. 

# ## 1. Building a Random Forest Model with All Features
# First, we will build an initial random forest model. This will work as our benchmark and we will use this to find out which features are contributing to the algorithm. The basic idea is that after we build a model, we will randomly shuffle on feature and caclulate the deviation from the model. If the new data (with one column shuffled) fits much better/worse than the old data, this feature is considered **important**.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics 
from IPython.display import display

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_table('../input/train.tsv', engine='c')
test = pd.read_table('../input/test.tsv', engine='c')

# Any results you write to the current directory are saved as output.


# In[ ]:


train['missing'] = train.brand_name.isnull()


# In[ ]:


train.brand_name[train['brand_name'].isnull()] = 'None'


# In[ ]:


train.dtypes


# In[ ]:


train['name'] = pd.Series(train.name, dtype="category").cat.codes
train['category_name'] = pd.Series(train.category_name, dtype="category").cat.codes
train['brand_name'] = pd.Series(train.brand_name, dtype="category").cat.codes
train['item_description'] = pd.Series(train.item_description, dtype="category").cat.codes
train['missing'] = pd.Series(train.missing, dtype="category").cat.codes


# In[ ]:


train.dtypes


# Split Training data into test and validation. Also separate dependent variable (price).

# In[ ]:


training = train.sample(frac=0.8,random_state=200)
validation = train.drop(training.index)


# In[ ]:


xtrain = training.drop('price', axis=1)
ytrain = training.price

xvalid = validation.drop('price', axis=1)
yvalid = validation.price


# In[ ]:


rf = RandomForestRegressor(n_jobs=-1, n_estimators=10)
rf.fit(xtrain, ytrain)


# In[ ]:


from sklearn import metrics
rf_score = rf.score(xtrain, ytrain)
rf_score


# ## 2. Computing Feature Importance

# We can first use a built-in object feature_importances. However, building our own feature importance algorithm will help us understand how these are computed.

# In[ ]:


feature_names = xtrain.columns


# In[ ]:


feature_imp = pd.DataFrame({'cols':feature_names, 'imp':rf.feature_importances_}).sort_values('imp', ascending=False)
feature_imp


# In[ ]:


feature_imp.plot('cols', 'imp', figsize=(10,6), legend=False);


# In[ ]:


xtrain_name = xtrain.copy()


# In[ ]:


xtrain_name['name'] = np.random.permutation(xtrain_name.name)


# In[ ]:


rf.score(xtrain_name, ytrain)


# In[ ]:


xtrain_item_description = xtrain.copy()
xtrain_item_description['item_description'] = np.random.permutation(xtrain_item_description.item_description)
rf.score(xtrain_item_description, ytrain)


# In[ ]:


xtrain_missing = xtrain.copy()
xtrain_missing['missing'] = np.random.permutation(xtrain_missing.missing)
rf.score(xtrain_missing, ytrain)


# We can see that the impact of the column/variable is directly shown through this algorithm. The more important feature it is, the more impact it has on the fitted score. For example, the feature name appeared to be the most imporatant feature in the built-in object and it clearly had the most impact in the fitted score when we randomly shuffled the name column. You can repeat this with all other variables to look for feature importance. It is important to note that we are using the **same random forest model** built on the orginial data (without any shuffling) to compute the fitted score. (we are **not** creating a new model with columns shuffled each time).
