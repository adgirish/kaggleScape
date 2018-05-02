
# coding: utf-8

# In[ ]:


#Script for faster calculation of Gini coefficient in python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


#The function used in most kernels
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


# In[ ]:


a = np.random.randint(0,2,100000)
p = np.random.rand(100000)
print(a[10:15], p[10:15])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'gini_normalized(a,p)')


# In[ ]:


#Remove redundant calls
def ginic(actual, pred):
    actual = np.asarray(actual) #In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalizedc(a, p):
    if p.ndim == 2:#Required for sklearn wrapper
        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1
    return ginic(a, p) / ginic(a, a)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'gini_normalizedc(a,p)')


# ### Wrappers for different algorithms

# In[ ]:


#XGBoost
from sklearn import metrics
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalizedc(labels, preds)
    return [('gini', gini_score)]

#LightGBM
def gini_lgb(actuals, preds):
    return 'gini', gini_normalizedc(actuals, preds), True

#SKlearn
gini_sklearn = metrics.make_scorer(gini_normalizedc, True, True)


# ### Cheers!

# ### Update:  sklearn example

# In[ ]:


train = pd.read_csv("../input/train.csv")
feats = [col for col in train.columns if col not in ['id','target']]

X = train[feats]
y = train['target']


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

#Initialize random forest
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20, max_features=0.2, n_jobs=-1)


# In[ ]:


#Stratified validation startegy
cv_1 = StratifiedKFold(n_splits=5, random_state=1).split(X, y)

#Check cross validation scores
cross_val_score(rfc, X, y, cv=cv_1, scoring=gini_sklearn, verbose=1, n_jobs=-1)


# ### Cheers again!!!
