
# coding: utf-8

# Another unsung [hero](https://github.com/fukatani/rgf_python) that you can use in your "stack-tastic" models

# In[ ]:


import gc
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold, cross_val_score
from rgf.sklearn import RGFClassifier
from sklearn.metrics import roc_auc_score


# In[ ]:


def ProjectOnMean(data1, data2, columnName):
    grpOutcomes = data1.groupby(list([columnName]))['target'].mean().reset_index()
    grpCount = data1.groupby(list([columnName]))['target'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.target
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['target'].values
    x = pd.merge(data2[[columnName, 'target']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=list([columnName]),
                 left_index=True)['target']

    
    return x.values

def GiniScore(y_actual, y_pred):
  return 2*roc_auc_score(y_actual, y_pred)-1


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
train.drop(unwanted,inplace=True,axis=1)
test.drop(unwanted,inplace=True,axis=1)
test.insert(1,'target',np.nan)


# In[ ]:


highcardinality =[]
for i in train.columns[1:-1]:
    if(((i.find('bin')!=-1) or (i.find('cat')!=-1))):
        highcardinality.append(i)

highcardinality


# In[ ]:


blindloodata = None
folds = 5
kf = StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(range(train.shape[0]),train.target)):
    print('Fold:',i)
    blindtrain = train.loc[test_index].copy() 
    vistrain = train.loc[train_index].copy()



    for c in highcardinality:
        blindtrain['loo'+c] = ProjectOnMean(vistrain,
                                            blindtrain,c)
    if(blindloodata is None):
        blindloodata = blindtrain.copy()
    else:
        blindloodata = pd.concat([blindloodata,blindtrain])

for c in highcardinality:
    test['loo'+c] = ProjectOnMean(train,
                                  test,c)
test.drop(highcardinality,inplace=True,axis=1)

train = blindloodata
train.drop(highcardinality,inplace=True,axis=1)
train = train.fillna(train.mean())
test = test.fillna(train.mean())


# In[ ]:


rgf = RGFClassifier(max_leaf=1000, #Try increasing this as a starter
                    algorithm="RGF_Sib",
                    test_interval=250,
                    loss="Log",
                    verbose=True)
rgf.fit(train[train.columns[2:]],train.target)
x = rgf.predict_proba(train[train.columns[2:]])
print(GiniScore(train.target,x[:,1]))


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
x = rgf.predict_proba(test[test.columns[2:]])
sub.target = x[:,1]
sub.to_csv('rgfsubmission.csv',index=False)

