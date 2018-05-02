
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import scikitplot.plotters as skplt
from sklearn.model_selection import StratifiedKFold


# In[3]:


dtype = {  'ip' : 'uint32',
           'app' : 'uint16',
           'device' : 'uint16',
           'os' : 'uint16',
           'channel' : 'uint8',
           'is_attributed' : 'uint8'}

usecol=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']


# In[4]:


df_train = pd.read_csv("../input/train.csv", dtype=dtype, infer_datetime_format=True, usecols=usecol, 
                               low_memory = True,nrows=20000000)
df_test = pd.read_csv("../input/test.csv")


# In[5]:


df_train.head()


# In[6]:


df_test.head()


# In[7]:


df_train['is_attributed'].value_counts()


# In[8]:


cols = ['ip', 'app', 'device', 'os', 'channel']
uniques_train = {col :df_train[col].nunique() for col in cols}
print('Train : Unique Values')
uniques_train


# In[9]:


cols = ['ip', 'app', 'device', 'os', 'channel']
uniques_test = {col :df_test[col].nunique() for col in cols}
print('Test : Unique Values')
uniques_test


# In[7]:


def mean_test_encoding(df_trn, df_tst, cols, target):    
   
    for col in cols:
        df_tst[col + '_mean_encoded'] = np.nan
        
    for col in cols:
        tr_mean = df_trn.groupby(col)[target].mean()
        mean = df_tst[col].map(tr_mean)
        df_tst[col + '_mean_encoded'] = mean

    prior = df_trn[target].mean()

    for col in cols:
        df_tst[col + '_mean_encoded'].fillna(prior, inplace = True) 
        
    return df_tst


# In[8]:


def mean_train_encoding(df, cols, target):
    y_tr = df[target].values
    skf = StratifiedKFold(5, shuffle = True, random_state=123)

    for col in cols:
        df[col + '_mean_encoded'] = np.nan

    for trn_ind , val_ind in skf.split(df,y_tr):
        x_tr, x_val = df.iloc[trn_ind], df.iloc[val_ind]

        for col in cols:
            tr_mean = x_tr.groupby(col)[target].mean()
            mean = x_val[col].map(tr_mean)
            df[col + '_mean_encoded'].iloc[val_ind] = mean

    prior = df[target].mean()

    for col in cols:
        df[col + '_mean_encoded'].fillna(prior, inplace = True) 
        
    return df


# In[9]:


y = df_train['is_attributed']
cols = ['app', 'channel']
target = 'is_attributed'
df_train = mean_train_encoding(df_train, cols, target)
df_test  = mean_test_encoding(df_train, df_test, cols, target)


# In[10]:


df_train.drop(['click_time','is_attributed'], axis = 1, inplace = True)   
df_test.drop(['click_time','click_id'], axis = 1, inplace = True)   


# In[14]:


def print_score(m, df, y):
    print('Accuracy: [Train , Val]')
    res =  [m.score(df, y)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)           
    print(res)
    
    print('Train Confusion Matrix')
    df_train_proba = m.predict_proba(df)
    df_train_pred_indices = np.argmax(df_train_proba, axis=1)
    classes_train = np.unique(y)
    preds_train = classes_train[df_train_pred_indices]    
    skplt.plot_confusion_matrix(y, preds_train)      


# In[15]:


df_train.head()


# In[12]:


test_submission = pd.read_csv("../input/sample_submission.csv")
test_submission.head()


# In[ ]:


clf = RandomForestClassifier(n_estimators=12, max_depth=6, min_samples_leaf=100, max_features=0.5, bootstrap=False, n_jobs=-1, random_state=123)
get_ipython().run_line_magic('time', 'clf.fit(df_train, y)')
print_score(clf, df_train, y)


# In[ ]:


cols = df_train.columns
Imp = clf.feature_importances_
feature_imp_dict = {}
for i in range(len(cols)):
    feature_imp_dict[cols[i]] = Imp[i]
print(feature_imp_dict)


# In[ ]:


y_pred = clf.predict_proba(df_test)
test_submission['is_attributed'] = y_pred[:,1]
test_submission.head()


# In[ ]:


test_submission.to_csv('submission_rf_.csv', index=False)

