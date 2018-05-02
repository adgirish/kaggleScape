
# coding: utf-8

# This is a kfold with both xgb and lgb. Some elements are from kernels by The1owl and HyungsukKang.
# 

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc

print('loading files...')
train = pd.read_csv('../input/train.csv', na_values=-1)
test = pd.read_csv('../input/test.csv', na_values=-1)
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  

for c in train.select_dtypes(include=['float64']).columns:
    train[c]=train[c].astype(np.float32)
    test[c]=test[c].astype(np.float32)
for c in train.select_dtypes(include=['int64']).columns[2:]:
    train[c]=train[c].astype(np.int8)
    test[c]=test[c].astype(np.int8)    

print(train.shape, test.shape)


# In[ ]:


# custom objective function (similar to auc)

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini(y, preds) / gini(y, y)
    return 'gini', score, True


# In[ ]:


# xgb
params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

X = train.drop(['id', 'target'], axis=1)
features = X.columns
X = X.values
y = train['target'].values
sub=test['id'].to_frame()
sub['target']=0

nrounds=200  # need to change to 2000
kfold = 2  # need to change to 5
skf = StratifiedKFold(n_splits=kfold, random_state=0)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    d_train = xgb.DMatrix(X_train, y_train) 
    d_valid = xgb.DMatrix(X_valid, y_valid) 
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                          feval=gini_xgb, maximize=True, verbose_eval=100)
    sub['target'] += xgb_model.predict(xgb.DMatrix(test[features].values), 
                        ntree_limit=xgb_model.best_ntree_limit+50) / (2*kfold)
gc.collect()
sub.head(2)

# lgb
params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':10, 'max_bin':10,  'objective': 'binary', 
          'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10,  'min_data': 500}

skf = StratifiedKFold(n_splits=kfold, random_state=1)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_eval = X[train_index], X[test_index]
    y_train, y_eval = y[train_index], y[test_index]
    lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds, 
                  lgb.Dataset(X_eval, label=y_eval), verbose_eval=100, 
                  feval=gini_lgb, early_stopping_rounds=100)
    sub['target'] += lgb_model.predict(test[features].values, 
                        num_iteration=lgb_model.best_iteration) / (2*kfold)
    
sub.to_csv('sub10.csv', index=False, float_format='%.5f') 
gc.collect()
sub.head(2)

