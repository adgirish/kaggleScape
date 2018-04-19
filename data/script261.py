
# coding: utf-8

# ### work with data

# In[ ]:


base_path = '../input/' # your folder

import pandas as pd
train=pd.read_csv(base_path + 'train.csv')
test=pd.read_csv(base_path + 'test.csv')

train['ps_ind_0609_bin'] = train.apply(lambda x: 1 if x['ps_ind_06_bin'] == 1 else (2 if x['ps_ind_07_bin'] == 1 else 
(
3 if x['ps_ind_08_bin'] == 1 else (4 if x['ps_ind_09_bin'] == 1 else 5)

)), axis = 1)

test['ps_ind_0609_bin'] = test.apply(lambda x: 1 if x['ps_ind_06_bin'] == 1 else (2 if x['ps_ind_07_bin'] == 1 else 
(
3 if x['ps_ind_08_bin'] == 1 else (4 if x['ps_ind_09_bin'] == 1 else 5)

)), axis = 1)

train.drop(['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin'], axis = 1, inplace = True)

test.drop(['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin'], axis = 1, inplace = True)

train['ps_car_13'] = (train['ps_car_13']*train['ps_car_13']* 48400).round(0)

test['ps_car_13'] = (test['ps_car_13']*test['ps_car_13']* 48400).round(0)

train['ps_car_12'] = (train['ps_car_12']*train['ps_car_12']).round(4) * 10000

test['ps_car_12'] = (test['ps_car_12']*test['ps_car_12']).round(4) * 10000

for c in train[[c for c in train.columns if 'bin' in c]].columns:
    for cc in train[[c for c in train.columns if 'bin' in c]].columns:
            if train[train[cc] * train[c] == 0].shape[0] == train.shape[0]:
                print(c, cc)

train['ps_ind_161718_bin'] = train.apply(lambda x: 1 if x['ps_ind_16_bin'] == 1 else
                                        (2 if x['ps_ind_17_bin'] == 1 else 3), axis = 1
                                        )

test['ps_ind_161718_bin'] = test.apply(lambda x: 1 if x['ps_ind_16_bin'] == 1 else
                                        (2 if x['ps_ind_17_bin'] == 1 else 3), axis = 1
                                        )

train.drop(['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin'], axis = 1, inplace = True)

test.drop(['ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin'], axis = 1, inplace = True)

train.to_csv(base_path + 'train_p.csv', index = False)

test.to_csv(base_path + 'test_p.csv', index = False)


# #  Set your folder

# In[ ]:


base_path = '../input/' # your folder


# # XGBOOST and LGB from kaggle kernel https://www.kaggle.com/rshally/porto-xgb-lgb-kfold-lb-0-282

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import gc

print('loading files...')
train = pd.read_csv(base_path+'train_p.csv', na_values=-1)
test = pd.read_csv(base_path+'test_p.csv', na_values=-1)
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

# xgb
params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

X = train.drop(['id', 'target'], axis=1)
features = X.columns
X = X.values
y = train['target'].values
sub=test['id'].to_frame()
sub['target']=0

sub_train = train['id'].to_frame()
sub_train['target']=0

nrounds=10**6  # need to change to 2000
kfold = 5  # need to change to 5
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
                        ntree_limit=xgb_model.best_ntree_limit+50) / (kfold)
    
    sub_train['target'] += xgb_model.predict(xgb.DMatrix(train[features].values), 
                        ntree_limit=xgb_model.best_ntree_limit+50) / (kfold)
    
gc.collect()
sub.head(2)

sub.to_csv(base_path+'test_sub_xgb.csv', index=False, float_format='%.5f')
sub_train.to_csv(base_path+'train_sub_xgb.csv', index=False, float_format='%.5f')


# lgb
sub['target']=0
sub_train['target']=0

params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':8, 'max_bin':10,  'objective': 'binary', 
          'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':5,  'min_data': 500}

skf = StratifiedKFold(n_splits=kfold, random_state=1)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
    X_train, X_eval = X[train_index], X[test_index]
    y_train, y_eval = y[train_index], y[test_index]
    lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), nrounds, 
                  lgb.Dataset(X_eval, label=y_eval), verbose_eval=100, 
                  feval=gini_lgb, early_stopping_rounds=100)
    sub['target'] += lgb_model.predict(test[features].values, 
                        num_iteration=lgb_model.best_iteration) / (kfold)
    sub_train['target'] += lgb_model.predict(train[features].values, 
                        num_iteration=lgb_model.best_iteration) / (kfold)
    
sub.to_csv(base_path+'test_sub_lgb.csv', index=False, float_format='%.5f') 
sub_train.to_csv(base_path+'train_sub_lgb.csv', index=False, float_format='%.5f')

gc.collect()
sub.head(2)


# ### Catboost

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import *
from catboost import CatBoostClassifier
from multiprocessing import *

train = pd.read_csv(base_path + 'train_p.csv')
test = pd.read_csv(base_path + 'test_p.csv')
col = [c for c in train.columns if c not in ['id','target']]
print(len(col))
col = [c for c in col if not c.startswith('ps_calc_')]
print(len(col))

train = train.replace(-1, np.NaN)
d_median = train.median(axis=0)
d_mean = train.mean(axis=0)
train = train.fillna(-1)

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            #df[c+str('_exp')] = np.exp(df[c].values) - 1
    
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    #p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    #p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) -1
    return g

def gini_catboost(pred, y):
    return gini(y, pred)

x1, x2, y1, y2 = model_selection.train_test_split(train, train['target'], test_size=0.25, random_state=99)

x1 = transform_df(x1)
x2 = transform_df(x2)
test = transform_df(test)
train = transform_df(train)

col = [c for c in x1.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]
print(x1.values.shape, x2.values.shape)

#remove duplicates just in case
#tdups = transform_df(train)
#dups = tdups[tdups.duplicated(subset=col, keep=False)]

#x1 = x1[~(x1['id'].isin(dups['id'].values))]
#x2 = x2[~(x2['id'].isin(dups['id'].values))]
#print(x1.values.shape, x2.values.shape)

y1 = x1['target']
y2 = x2['target']
x1 = x1[col]
x2 = x2[col]

model3 = CatBoostClassifier(iterations=1200, learning_rate=0.02, depth=7, loss_function='Logloss', eval_metric='AUC', random_seed=99, od_type='Iter', od_wait=100) 
model3.fit(x1[col], y1, eval_set=(x2[col], y2), use_best_model=True, verbose=True)
print(gini_catboost(model3.predict_proba(x2[col])[:,1], y2))
test['target'] = model3.predict_proba(test[col])[:,1]
test['target'] = (np.exp(test['target'].values) - 1.0).clip(0,1)
train['target'] = model3.predict_proba(train[col])[:,1]
train['target'] = (np.exp(train['target'].values) - 1.0).clip(0,1)
test[['id','target']].to_csv(base_path + 'test_catboost_submission.csv', index=False, float_format='%.5f')
train[['id','target']].to_csv(base_path + 'train_catboost_submission.csv', index=False, float_format='%.5f')

#Extras
import matplotlib.pyplot as plt

df = pd.DataFrame({'imp': model3.feature_importances_, 'col':col})
df = df.sort_values(['imp','col'], ascending=[True, False])
_ = df.plot(kind='barh', x='col', y='imp', figsize=(7,12))
plt.savefig('catboost_feature_importance.png')


# # xgboost upsampled https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

"""
This simple scripts demonstrates the use of xgboost eval results to get the best round
for the current fold and accross folds. 
It also shows an upsampling method that limits cross-validation overfitting.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import gc
from numba import jit
from sklearn.preprocessing import LabelEncoder
import time 
import xgboost as xgb


@jit
def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
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

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = eval_gini(labels, preds)
    return [('gini', gini_score)]


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)

gc.enable()

trn_df = pd.read_csv(base_path + "train_p.csv", index_col=0)
sub_df = pd.read_csv(base_path + "test_p.csv", index_col=0)

target = trn_df["target"]
del trn_df["target"]

train_features = [
    "ps_car_13",  #            : 1571.65 / shadow  609.23
"ps_reg_03",  #            : 1408.42 / shadow  511.15
"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72
"ps_ind_03",  #            : 1219.47 / shadow  230.55
"ps_ind_15",  #            :  922.18 / shadow  242.00
"ps_reg_02",  #            :  920.65 / shadow  267.50
"ps_car_14",  #            :  798.48 / shadow  549.58
"ps_car_12",  #            :  731.93 / shadow  293.62
"ps_car_01_cat",  #        :  698.07 / shadow  178.72
"ps_car_07_cat",  #        :  694.53 / shadow   36.35
"ps_car_03_cat",  #        :  611.73 / shadow   50.67
"ps_reg_01",  #            :  598.60 / shadow  178.57
"ps_car_15",  #            :  593.35 / shadow  226.43
"ps_ind_01",  #            :  547.32 / shadow  154.58
"ps_ind_161718_bin",  #        :  475.37 / shadow   34.17
"ps_ind_0609_bin",  #        :  435.28 / shadow   28.92
"ps_car_06_cat",  #        :  398.02 / shadow  212.43
"ps_car_04_cat",  #        :  376.87 / shadow   76.98
"ps_car_09_cat",  #        :  214.12 / shadow   81.38
"ps_car_02_cat",  #        :  203.03 / shadow   26.67
"ps_ind_02_cat",  #        :  189.47 / shadow   65.68
"ps_car_11",  #            :  173.28 / shadow   76.45
"ps_car_05_cat",  #        :  172.75 / shadow   62.92
"ps_calc_09",  #           :  169.13 / shadow  129.72
"ps_calc_05",  #           :  148.83 / shadow  120.68
"ps_car_08_cat",  #        :  120.87 / shadow   28.82
"ps_ind_04_cat",  #        :  107.27 / shadow   37.43
"ps_ind_12_bin",  #        :   39.67 / shadow   15.52
"ps_ind_14",  #            :   37.37 / shadow   16.65
]
# add combinations
combs = [
    ('ps_reg_01', 'ps_car_02_cat'),  
    ('ps_reg_01', 'ps_car_04_cat'),
]
start = time.time()
for n_c, (f1, f2) in enumerate(combs):
    name1 = f1 + "_plus_" + f2
    print('current feature %60s %4d in %5.1f'
          % (name1, n_c + 1, (time.time() - start) / 60), end='')
    print('\r' * 75, end='')
    trn_df[name1] = trn_df[f1].apply(lambda x: str(x)) + "_" + trn_df[f2].apply(lambda x: str(x))
    sub_df[name1] = sub_df[f1].apply(lambda x: str(x)) + "_" + sub_df[f2].apply(lambda x: str(x))
    # Label Encode
    lbl = LabelEncoder()
    lbl.fit(list(trn_df[name1].values) + list(sub_df[name1].values))
    trn_df[name1] = lbl.transform(list(trn_df[name1].values))
    sub_df[name1] = lbl.transform(list(sub_df[name1].values))

    train_features.append(name1)
    
trn_df = trn_df[train_features]
sub_df = sub_df[train_features]

f_cats = [f for f in trn_df.columns if "_cat" in f]

for f in f_cats:
    trn_df[f + "_avg"], sub_df[f + "_avg"] = target_encode(trn_series=trn_df[f],
                                         tst_series=sub_df[f],
                                         target=target,
                                         min_samples_leaf=200,
                                         smoothing=10,
                                         noise_level=0)

n_splits = 5
n_estimators = 200
folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=15) 
imp_df = np.zeros((len(trn_df.columns), n_splits))
xgb_evals = np.zeros((n_estimators, n_splits))
oof = np.empty(len(trn_df))
sub_preds = np.zeros(len(sub_df))
sub_preds_train = np.zeros(len(trn_df))
increase = True
np.random.seed(0)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
    trn_dat, trn_tgt = trn_df.iloc[trn_idx], target.iloc[trn_idx]
    val_dat, val_tgt = trn_df.iloc[val_idx], target.iloc[val_idx]
    
   


    params = {'n_estimators':n_estimators,
                        'max_depth':4,
                        'objective':"binary:logistic",
                        'learning_rate':.1, 
                        'subsample':.8, 
                        'colsample_bytree':.8,
                        'gamma':1,
                        'reg_alpha':0,
                        'reg_lambda':1,
                        'nthread':2}
    # Upsample during cross validation to avoid having the same samples
    # in both train and validation sets
    # Validation set is not up-sampled to monitor overfitting
    if increase:
        # Get positive examples
        pos = pd.Series(trn_tgt == 1)
        # Add positive examples
        trn_dat = pd.concat([trn_dat, trn_dat.loc[pos]], axis=0)
        trn_tgt = pd.concat([trn_tgt, trn_tgt.loc[pos]], axis=0)
        # Shuffle data
        idx = np.arange(len(trn_dat))
        np.random.shuffle(idx)
        trn_dat = trn_dat.iloc[idx]
        trn_tgt = trn_tgt.iloc[idx]
        
    d_train = xgb.DMatrix(trn_dat, trn_tgt) 
    d_valid = xgb.DMatrix(val_dat, val_tgt) 
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        
    clf = xgb.train(params, d_train, 10**6, watchlist, early_stopping_rounds=20, 
                          feval=gini_xgb, maximize=True, verbose_eval=100)
            

    sub_preds += clf.predict(xgb.DMatrix(sub_df), ntree_limit=clf.best_ntree_limit) / n_splits
    sub_preds_train += clf.predict(xgb.DMatrix(trn_df), ntree_limit=clf.best_ntree_limit) / n_splits

    # Display results
#     print("Fold %2d : %.6f @%4d / best score is %.6f @%4d"
#           % (fold_ + 1,
#              eval_gini(val_tgt, oof[val_idx]),
#              n_estimators,
#              xgb_evals[best_round, fold_],
#              best_round))
          
print("Full OOF score : %.6f" % eval_gini(target, oof))

# Compute mean score and std
mean_eval = np.mean(xgb_evals, axis=1)
std_eval = np.std(xgb_evals, axis=1)
best_round = np.argsort(mean_eval)[::-1][0]

print("Best mean score : %.6f + %.6f @%4d"
      % (mean_eval[best_round], std_eval[best_round], best_round))
    
importances = sorted([(trn_df.columns[i], imp) for i, imp in enumerate(imp_df.mean(axis=1))],
                     key=lambda x: x[1])

for f, imp in importances[::-1]:
    print("%-34s : %10.4f" % (f, imp))
    
sub_df["target"] = sub_preds
trn_df["target"] = sub_preds_train

sub_df[["target"]].to_csv(base_path + "test_submission.csv", index=True, float_format="%.9f")
trn_df[["target"]].to_csv(base_path + "train_submission.csv", index=True, float_format="%.9f")


# # GP https://www.kaggle.com/scirpus/big-gp/code

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score



def GiniScore(y_actual, y_pred):
  return 2*roc_auc_score(y_actual, y_pred)-1


def Outputs(p):
    return 1./(1.+np.exp(-p))


def GPI(data):
    v = pd.DataFrame()
    v["0"] = -3.274750
    v["1"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] + (data["ps_reg_01"] + (data["ps_car_12"] + (data["loo_ps_car_03_cat"] + data["loo_ps_car_07_cat"])))))
    v["2"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] + ((data["ps_reg_03"] + data["ps_car_12"]) + (data["loo_ps_ind_06_bin"] + data["loo_ps_ind_05_cat"]))))
    v["3"] = 0.020000*np.tanh((data["loo_ps_car_01_cat"] + (data["loo_ps_ind_17_bin"] + ((data["ps_car_12"] + data["ps_reg_03"]) + data["loo_ps_ind_07_bin"]))))
    v["4"] = 0.020000*np.tanh((data["loo_ps_car_01_cat"] + ((data["loo_ps_car_04_cat"] + (data["ps_reg_03"] + data["ps_car_15"])) * 3.0)))
    v["5"] = 0.020000*np.tanh(((data["ps_car_13"] + ((data["loo_ps_car_05_cat"] + (data["ps_reg_01"] + data["loo_ps_car_09_cat"]))/2.0)) * (10.28825187683105469)))
    v["6"] = 0.020000*np.tanh(((8.0) * (data["loo_ps_car_04_cat"] + ((data["loo_ps_car_01_cat"] + data["loo_ps_ind_05_cat"]) + data["ps_car_15"]))))
    v["7"] = 0.020000*np.tanh(((data["ps_car_13"] + (((data["loo_ps_car_01_cat"] + data["ps_reg_03"]) + data["loo_ps_ind_16_bin"])/2.0)) * 8.428570))
    v["8"] = 0.020000*np.tanh(((10.86397266387939453) * (((data["ps_reg_03"] + data["loo_ps_car_07_cat"]) + data["loo_ps_ind_06_bin"]) + data["loo_ps_car_11_cat"])))
    v["9"] = 0.020000*np.tanh((data["ps_reg_03"] + (data["loo_ps_ind_16_bin"] + (data["loo_ps_ind_07_bin"] + (data["ps_car_13"] + data["loo_ps_car_03_cat"])))))
    v["10"] = 0.020000*np.tanh((((data["ps_car_13"] + data["ps_reg_02"]) - data["ps_ind_15"]) + (data["loo_ps_car_11_cat"] + data["loo_ps_ind_17_bin"])))
    v["11"] = 0.020000*np.tanh(((data["ps_car_13"] + data["ps_reg_02"]) + (data["loo_ps_ind_07_bin"] + (data["loo_ps_ind_17_bin"] + data["loo_ps_car_01_cat"]))))
    v["12"] = 0.020000*np.tanh(((data["ps_car_13"] + (data["ps_reg_02"] + ((data["loo_ps_car_09_cat"] + data["loo_ps_ind_16_bin"])/2.0))) * 8.428570))
    v["13"] = 0.020000*np.tanh((data["ps_reg_02"] + (data["loo_ps_ind_08_bin"] + ((data["loo_ps_ind_16_bin"] + data["ps_car_13"]) + data["loo_ps_ind_07_bin"]))))
    v["14"] = 0.020000*np.tanh((29.500000 * (((data["ps_car_13"] + (data["loo_ps_car_09_cat"] + data["loo_ps_ind_16_bin"]))/2.0) + data["loo_ps_ind_05_cat"])))
    v["15"] = 0.020000*np.tanh((29.500000 * ((data["loo_ps_car_04_cat"] + (data["loo_ps_car_03_cat"] + data["ps_car_13"])) + 0.945455)))
    v["16"] = 0.020000*np.tanh((8.428570 * ((data["loo_ps_ind_05_cat"] + data["loo_ps_car_06_cat"]) + (data["loo_ps_car_07_cat"] + data["loo_ps_ind_17_bin"]))))
    v["17"] = 0.020000*np.tanh((((0.633333 + (data["loo_ps_ind_17_bin"] + data["ps_car_13"])) + data["ps_reg_03"]) * 29.500000))
    v["18"] = 0.020000*np.tanh((((data["loo_ps_ind_17_bin"] + data["loo_ps_car_11_cat"]) + (data["loo_ps_car_07_cat"] - data["ps_ind_15"])) + data["ps_ind_03"]))
    v["19"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + (data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_09_bin"] + data["loo_ps_ind_06_bin"]))) * (4.85490655899047852)))
    v["20"] = 0.020000*np.tanh(((((data["loo_ps_ind_06_bin"] + data["loo_ps_car_11_cat"]) + data["loo_ps_car_07_cat"]) + data["loo_ps_car_09_cat"]) - data["ps_ind_15"]))
    v["21"] = 0.020000*np.tanh((8.428570 * ((data["loo_ps_ind_05_cat"] + data["ps_car_13"]) + ((data["loo_ps_car_05_cat"] + data["loo_ps_car_01_cat"])/2.0))))
    v["22"] = 0.020000*np.tanh((((data["ps_ind_03"] + (data["loo_ps_ind_16_bin"] + (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_06_bin"])))/2.0) * 29.500000))
    v["23"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_09_bin"] + (data["ps_reg_03"] + data["loo_ps_ind_05_cat"]))) * 29.500000))
    v["24"] = 0.020000*np.tanh((data["loo_ps_car_03_cat"] - (data["ps_ind_15"] - (data["loo_ps_ind_05_cat"] + ((data["loo_ps_ind_07_bin"] + data["loo_ps_ind_17_bin"])/2.0)))))
    v["25"] = 0.020000*np.tanh(((data["ps_reg_02"] + ((data["loo_ps_car_07_cat"] - -1.0) + data["ps_car_13"])) * (13.29769420623779297)))
    v["26"] = 0.020000*np.tanh((29.500000 * ((((data["loo_ps_car_01_cat"] + data["ps_car_13"])/2.0) + (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"]))/2.0)))
    v["27"] = 0.020000*np.tanh((data["ps_reg_03"] + (1.480000 + (data["loo_ps_ind_06_bin"] + (data["loo_ps_car_11_cat"] + data["loo_ps_car_01_cat"])))))
    v["28"] = 0.020000*np.tanh((data["loo_ps_car_11_cat"] + ((data["loo_ps_ind_05_cat"] + (data["loo_ps_car_09_cat"] - data["ps_ind_15"])) + data["loo_ps_car_03_cat"])))
    v["29"] = 0.020000*np.tanh(((10.24501132965087891) * (data["loo_ps_ind_05_cat"] + (data["ps_ind_03"] + (data["loo_ps_ind_09_bin"] - data["ps_ind_15"])))))
    v["30"] = 0.020000*np.tanh(((data["loo_ps_car_11_cat"] + (data["loo_ps_ind_05_cat"] + data["loo_ps_car_07_cat"])) + (data["loo_ps_car_09_cat"] - data["missing"])))
    v["31"] = 0.020000*np.tanh((29.500000 * (data["loo_ps_car_01_cat"] + ((data["loo_ps_car_07_cat"] + data["loo_ps_ind_05_cat"]) + data["loo_ps_ind_16_bin"]))))
    v["32"] = 0.020000*np.tanh(((data["ps_reg_03"] + ((data["ps_ind_01"] + 0.887097) + data["loo_ps_car_09_cat"])) * (10.0)))
    v["33"] = 0.020000*np.tanh(((data["loo_ps_car_03_cat"] + (data["loo_ps_car_03_cat"] - (data["ps_ind_15"] - 1.480000))) * 29.500000))
    v["34"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] + ((14.32789230346679688) * (data["ps_ind_03"] + (data["ps_ind_03"] * data["ps_ind_03"])))))
    v["35"] = 0.020000*np.tanh((29.500000 * (data["loo_ps_ind_17_bin"] + (((data["ps_car_13"] + data["ps_ind_03"])/2.0) + data["loo_ps_ind_05_cat"]))))
    v["36"] = 0.020000*np.tanh(((9.0) * (data["ps_reg_02"] + (data["loo_ps_car_07_cat"] + ((data["ps_car_15"] + 1.0)/2.0)))))
    v["37"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_08_bin"] + ((data["loo_ps_ind_16_bin"] + data["loo_ps_ind_07_bin"])/2.0))) * 8.428570))
    v["38"] = 0.020000*np.tanh(((2.0 * ((data["loo_ps_car_03_cat"] + data["loo_ps_car_07_cat"]) + data["loo_ps_ind_05_cat"])) + data["loo_ps_car_04_cat"]))
    v["39"] = 0.020000*np.tanh((29.500000 * (data["loo_ps_ind_07_bin"] + ((data["loo_ps_car_07_cat"] + 1.089890) - data["ps_ind_15"]))))
    v["40"] = 0.020000*np.tanh(((10.18701076507568359) * ((data["ps_ind_03"] * data["ps_ind_03"]) + (data["loo_ps_ind_05_cat"] + data["ps_ind_03"]))))
    v["41"] = 0.020000*np.tanh((8.428570 * (data["loo_ps_ind_02_cat"] + (data["ps_car_13"] + (data["loo_ps_car_07_cat"] + data["loo_ps_ind_09_bin"])))))
    v["42"] = 0.020000*np.tanh((8.428570 * (((data["loo_ps_car_01_cat"] - 0.435484) + data["loo_ps_ind_05_cat"]) + data["loo_ps_ind_17_bin"])))
    v["43"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] + (29.500000 * (data["ps_ind_03"] + (data["ps_ind_03"] * data["ps_ind_03"])))))
    v["44"] = 0.020000*np.tanh((8.428570 * ((data["loo_ps_ind_05_cat"] + data["loo_ps_car_09_cat"]) - (data["ps_ind_15"] + data["loo_ps_ind_18_bin"]))))
    v["45"] = 0.020000*np.tanh((data["ps_ind_01"] + ((5.76565647125244141) * (data["loo_ps_ind_07_bin"] + (0.887097 - data["ps_ind_15"])))))
    v["46"] = 0.020000*np.tanh(((7.0) * (2.352940 * ((data["ps_ind_03"] * data["ps_ind_03"]) + data["ps_ind_03"]))))
    v["47"] = 0.020000*np.tanh((29.500000 * ((data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_17_bin"] + data["loo_ps_ind_05_cat"])) - data["ps_ind_03"])))
    v["48"] = 0.020000*np.tanh(((9.90538215637207031) * (((data["loo_ps_ind_06_bin"] + data["loo_ps_car_07_cat"]) + data["loo_ps_ind_09_bin"]) + data["loo_ps_ind_09_bin"])))
    v["49"] = 0.020000*np.tanh((data["ps_ind_01"] + ((data["loo_ps_car_01_cat"] + (data["ps_ind_01"] + data["loo_ps_ind_06_bin"])) + data["loo_ps_car_09_cat"])))
    v["50"] = 0.020000*np.tanh((29.500000 * (data["loo_ps_ind_02_cat"] + (data["loo_ps_car_09_cat"] + ((data["loo_ps_car_07_cat"] + data["loo_ps_ind_05_cat"])/2.0)))))
    v["51"] = 0.020000*np.tanh((29.500000 * (data["ps_car_13"] + (data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_06_bin"] - 2.352940)))))
    v["52"] = 0.020000*np.tanh((29.500000 * (data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_07_bin"] * data["ps_ind_03"])))))
    v["53"] = 0.020000*np.tanh((29.500000 * (((9.0) * ((data["loo_ps_ind_04_cat"] + data["ps_reg_03"])/2.0)) - data["ps_ind_15"])))
    v["54"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_17_bin"] + (data["ps_reg_02"] * data["loo_ps_ind_05_cat"])))))
    v["55"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] + ((data["loo_ps_ind_06_bin"] * (data["ps_ind_03"] + data["loo_ps_ind_05_cat"])) - data["ps_car_11"])))
    v["56"] = 0.020000*np.tanh(((data["ps_ind_01"] + (data["ps_ind_03"] + data["loo_ps_ind_05_cat"])) * (data["loo_ps_ind_07_bin"] + data["loo_ps_car_05_cat"])))
    v["57"] = 0.020000*np.tanh(((data["loo_ps_ind_16_bin"] * data["loo_ps_car_03_cat"]) - (data["ps_ind_15"] - ((data["ps_reg_02"] + data["loo_ps_ind_02_cat"])/2.0))))
    v["58"] = 0.020000*np.tanh((((data["ps_car_13"] + -2.0) + (data["loo_ps_ind_05_cat"] - data["ps_ind_15"])) - data["ps_car_11"]))
    v["59"] = 0.020000*np.tanh(((data["ps_car_15"] + (3.0 * data["loo_ps_ind_17_bin"])) + (data["ps_ind_03"] * data["ps_ind_03"])))
    v["60"] = 0.020000*np.tanh((29.500000 * ((data["ps_ind_03"] * data["ps_ind_03"]) + data["ps_ind_03"])))
    v["61"] = 0.020000*np.tanh((((data["loo_ps_ind_17_bin"] + data["missing"]) + (data["loo_ps_car_11_cat"] + data["loo_ps_ind_17_bin"])) * data["loo_ps_car_05_cat"]))
    v["62"] = 0.020000*np.tanh((8.428570 + (((data["loo_ps_car_03_cat"] * data["loo_ps_ind_09_bin"]) + data["loo_ps_car_01_cat"]) * 29.500000)))
    v["63"] = 0.020000*np.tanh(((data["ps_ind_01"] + (data["loo_ps_car_11_cat"] + data["loo_ps_car_05_cat"])) * (data["loo_ps_ind_05_cat"] - data["ps_car_11"])))
    v["64"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] + (data["ps_ind_01"] + (data["loo_ps_ind_09_bin"] * (data["ps_ind_03"] + data["ps_ind_01"])))))
    v["65"] = 0.020000*np.tanh(((data["loo_ps_car_11_cat"] + data["loo_ps_ind_02_cat"]) + (data["loo_ps_car_05_cat"] * (data["loo_ps_ind_09_bin"] + data["loo_ps_ind_06_bin"]))))
    v["66"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_06_bin"]) * 29.500000) + (data["loo_ps_ind_02_cat"] * data["loo_ps_ind_05_cat"])))
    v["67"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - data["ps_reg_01"]) * ((data["ps_ind_01"] + data["loo_ps_ind_06_bin"]) + data["loo_ps_car_11_cat"])))
    v["68"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - data["ps_car_11"]) * (3.0 * (data["loo_ps_ind_17_bin"] + data["loo_ps_car_04_cat"]))))
    v["69"] = 0.019996*np.tanh((((data["ps_reg_02"] - data["ps_car_11"]) + (data["loo_ps_ind_04_cat"] * 3.0)) - data["ps_ind_15"]))
    v["70"] = 0.020000*np.tanh((-((data["loo_ps_ind_02_cat"] * (data["ps_ind_03"] + (data["ps_ind_03"] + data["ps_ind_03"]))))))
    v["71"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] + (data["missing"] + ((8.37885093688964844) * (data["loo_ps_ind_17_bin"] + data["loo_ps_ind_02_cat"])))))
    v["72"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] + (data["ps_reg_03"] + (1.480000 * (data["loo_ps_car_04_cat"] * data["ps_ind_01"])))))
    v["73"] = 0.020000*np.tanh(((((5.0) * data["loo_ps_ind_05_cat"]) + data["loo_ps_car_09_cat"]) * (data["loo_ps_ind_17_bin"] + data["ps_reg_03"])))
    v["74"] = 0.020000*np.tanh(((((data["loo_ps_ind_07_bin"] + data["loo_ps_car_01_cat"])/2.0) * data["loo_ps_ind_16_bin"]) + (data["ps_ind_15"] * data["loo_ps_ind_18_bin"])))
    v["75"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] * data["loo_ps_car_01_cat"]) + data["loo_ps_car_05_cat"]) * (data["loo_ps_ind_05_cat"] - data["ps_ind_15"])))
    v["76"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * ((data["loo_ps_car_04_cat"] + data["loo_ps_ind_05_cat"]) + (data["loo_ps_car_07_cat"] + data["ps_ind_03"]))))
    v["77"] = 0.020000*np.tanh(((data["loo_ps_car_05_cat"] + data["loo_ps_car_03_cat"]) * (data["ps_ind_03"] + (data["missing"] + data["loo_ps_ind_05_cat"]))))
    v["78"] = 0.020000*np.tanh(((data["ps_ind_01"] * data["loo_ps_car_03_cat"]) + (data["loo_ps_car_08_cat"] + (data["loo_ps_ind_17_bin"] * data["loo_ps_car_03_cat"]))))
    v["79"] = 0.020000*np.tanh((data["ps_ind_03"] * (3.0 * (3.0 * (-(data["loo_ps_ind_02_cat"]))))))
    v["80"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"]) * (data["loo_ps_car_09_cat"] + data["ps_reg_03"])) - data["loo_ps_ind_16_bin"]))
    v["81"] = 0.020000*np.tanh((data["loo_ps_car_03_cat"] * ((data["loo_ps_ind_09_bin"] - data["ps_car_15"]) + (data["loo_ps_ind_02_cat"] - data["loo_ps_ind_18_bin"]))))
    v["82"] = 0.020000*np.tanh((((data["ps_ind_01"] + data["loo_ps_ind_06_bin"]) * (data["loo_ps_ind_16_bin"] - data["ps_reg_01"])) + data["loo_ps_car_07_cat"]))
    v["83"] = 0.020000*np.tanh((data["ps_ind_15"] * (data["loo_ps_ind_16_bin"] - ((data["loo_ps_car_11_cat"] + data["missing"]) + data["loo_ps_ind_17_bin"]))))
    v["84"] = 0.020000*np.tanh((((data["loo_ps_ind_07_bin"] + data["loo_ps_ind_02_cat"])/2.0) + (data["ps_ind_01"] * (0.600000 - data["ps_ind_15"]))))
    v["85"] = 0.020000*np.tanh(((data["ps_ind_01"] + data["loo_ps_car_04_cat"]) * ((data["loo_ps_ind_05_cat"] - data["ps_ind_15"]) + -1.0)))
    v["86"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] + (data["ps_reg_03"] + (data["ps_reg_03"] + (data["loo_ps_ind_17_bin"] * data["loo_ps_car_07_cat"])))))
    v["87"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((((4.0) * data["loo_ps_ind_17_bin"]) + data["loo_ps_car_06_cat"]) - data["ps_reg_01"])))
    v["88"] = 0.020000*np.tanh(((-1.0 + data["loo_ps_ind_12_bin"]) - (data["loo_ps_car_10_cat"] + (data["ps_car_11"] - data["loo_ps_car_09_cat"]))))
    v["89"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((data["loo_ps_car_09_cat"] + ((-1.0 + data["ps_reg_03"])/2.0)) + data["ps_reg_03"])))
    v["90"] = 0.020000*np.tanh((((data["loo_ps_ind_09_bin"] + data["loo_ps_ind_05_cat"]) * ((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"])/2.0)) - data["loo_ps_ind_05_cat"]))
    v["91"] = 0.020000*np.tanh((-(((data["loo_ps_ind_02_cat"] + data["loo_ps_ind_02_cat"]) * (data["ps_ind_03"] + data["loo_ps_ind_06_bin"])))))
    v["92"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] + (data["ps_reg_03"] * ((-(data["ps_reg_01"])) + data["loo_ps_ind_17_bin"]))))
    v["93"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] * 29.500000) * (data["loo_ps_car_08_cat"] + (data["loo_ps_car_08_cat"] - data["ps_ind_03"]))))
    v["94"] = 0.020000*np.tanh((((data["loo_ps_car_03_cat"] * data["loo_ps_car_09_cat"]) * (data["loo_ps_car_04_cat"] * data["loo_ps_car_04_cat"])) - data["loo_ps_car_04_cat"]))
    v["95"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + -1.0) - (data["ps_car_12"] * (2.0 * data["ps_car_11"]))))
    v["96"] = 0.020000*np.tanh((-((data["ps_reg_03"] * ((data["loo_ps_car_01_cat"] - data["loo_ps_car_03_cat"]) + data["loo_ps_ind_08_bin"])))))
    v["97"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * (data["loo_ps_car_07_cat"] + (((data["loo_ps_ind_07_bin"] + data["ps_car_12"])/2.0) - data["ps_reg_01"]))))
    v["98"] = 0.020000*np.tanh(((((data["loo_ps_ind_07_bin"] + data["ps_ind_15"])/2.0) + data["loo_ps_ind_17_bin"]) * (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"])))
    v["99"] = 0.019992*np.tanh(((data["loo_ps_ind_05_cat"] * data["loo_ps_car_09_cat"]) + (-((data["ps_reg_01"] * data["ps_ind_03"])))))
    v["100"] = 0.019988*np.tanh((data["loo_ps_ind_04_cat"] + (data["loo_ps_car_05_cat"] * (data["ps_ind_01"] + (data["ps_ind_01"] + data["loo_ps_car_01_cat"])))))
    v["101"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_05_cat"] * np.tanh((data["loo_ps_ind_05_cat"] + (-(2.352940)))))))
    v["102"] = 0.020000*np.tanh((-((data["ps_reg_01"] * (data["loo_ps_car_09_cat"] + (data["ps_reg_03"] + data["loo_ps_ind_05_cat"]))))))
    v["103"] = 0.020000*np.tanh(((-1.0 + (data["loo_ps_car_09_cat"] + ((data["ps_reg_01"] * data["ps_reg_01"]) + data["ps_reg_01"])))/2.0))
    v["104"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] + ((data["loo_ps_ind_02_cat"] + -2.0)/2.0)) * (data["loo_ps_ind_05_cat"] - data["ps_car_15"])))
    v["105"] = 0.019992*np.tanh((((data["loo_ps_ind_17_bin"] + data["loo_ps_car_07_cat"]) + data["loo_ps_ind_17_bin"]) * ((data["ps_reg_02"] + data["loo_ps_car_09_cat"])/2.0)))
    v["106"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_07_bin"] + (data["loo_ps_ind_07_bin"] + (data["loo_ps_car_09_cat"] + -1.0)))))
    v["107"] = 0.020000*np.tanh(((data["ps_car_15"] + (data["loo_ps_ind_07_bin"] * data["loo_ps_ind_17_bin"])) - (data["ps_reg_03"] * data["loo_ps_car_01_cat"])))
    v["108"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] * ((data["loo_ps_ind_04_cat"] - (data["loo_ps_ind_06_bin"] - 0.432099)) - data["loo_ps_car_01_cat"])))
    v["109"] = 0.019988*np.tanh((data["loo_ps_ind_02_cat"] - (data["ps_ind_01"] * (-2.0 + (data["ps_ind_01"] * data["ps_ind_01"])))))
    v["110"] = 0.020000*np.tanh(((29.500000 + (data["ps_car_15"] * 29.500000)) * (0.791667 - data["ps_car_15"])))
    v["111"] = 0.020000*np.tanh(((((data["loo_ps_ind_04_cat"] + -1.0) + data["loo_ps_car_08_cat"])/2.0) + (data["loo_ps_ind_05_cat"] * data["ps_reg_03"])))
    v["112"] = 0.020000*np.tanh((data["loo_ps_car_01_cat"] * ((data["ps_reg_03"] * ((data["loo_ps_car_05_cat"] + data["ps_reg_03"])/2.0)) - data["ps_car_15"])))
    v["113"] = 0.020000*np.tanh((data["ps_reg_02"] * ((data["loo_ps_ind_17_bin"] - data["loo_ps_car_01_cat"]) - (data["ps_ind_01"] - data["loo_ps_ind_17_bin"]))))
    v["114"] = 0.020000*np.tanh((data["ps_ind_03"] * (data["ps_reg_03"] - (1.480000 + (data["ps_ind_15"] - data["ps_reg_03"])))))
    v["115"] = 0.019988*np.tanh(((((data["loo_ps_car_01_cat"] + (data["ps_car_11"] * data["ps_car_11"]))/2.0) + (data["ps_reg_01"] * data["missing"]))/2.0))
    v["116"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] * ((data["loo_ps_ind_04_cat"] - data["ps_ind_01"]) + data["ps_ind_03"])) - data["loo_ps_ind_02_cat"]))
    v["117"] = 0.020000*np.tanh((data["loo_ps_car_04_cat"] * ((data["loo_ps_car_09_cat"] - data["ps_car_11"]) - ((data["loo_ps_car_06_cat"] + data["loo_ps_car_04_cat"])/2.0))))
    v["118"] = 0.020000*np.tanh((((data["loo_ps_ind_04_cat"] + data["ps_ind_03"])/2.0) * ((data["loo_ps_car_02_cat"] - data["ps_reg_01"]) + data["loo_ps_car_07_cat"])))
    v["119"] = 0.020000*np.tanh(((((data["ps_car_15"] * (-(data["loo_ps_car_03_cat"]))) + data["loo_ps_car_01_cat"])/2.0) - data["loo_ps_car_07_cat"]))
    v["120"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((data["loo_ps_ind_05_cat"] * (data["loo_ps_car_03_cat"] * data["loo_ps_car_11_cat"])) - 0.633333)))
    v["121"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] + data["ps_car_12"]) + data["loo_ps_ind_18_bin"])/2.0) * ((data["missing"] + data["ps_reg_02"])/2.0)))
    v["122"] = 0.020000*np.tanh(((((data["ps_ind_03"] + data["loo_ps_ind_04_cat"])/2.0) * ((data["ps_ind_03"] + data["loo_ps_ind_04_cat"])/2.0)) - 0.432099))
    v["123"] = 0.020000*np.tanh(((data["ps_reg_03"] * data["ps_reg_03"]) * (((data["loo_ps_ind_05_cat"] - data["ps_reg_03"]) + data["ps_car_12"])/2.0)))
    v["124"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (data["ps_car_13"] + ((data["loo_ps_car_04_cat"] - data["ps_ind_03"]) - 1.089890))))
    v["125"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + -2.0)/2.0) * (data["loo_ps_ind_02_cat"] + (data["ps_car_11"] * data["ps_ind_01"]))))
    v["126"] = 0.019988*np.tanh(((data["loo_ps_ind_02_cat"] + ((data["loo_ps_car_07_cat"] * data["loo_ps_ind_08_bin"]) - (data["ps_ind_15"] * data["loo_ps_car_06_cat"])))/2.0))
    v["127"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] * (data["loo_ps_car_09_cat"] * np.tanh(((data["loo_ps_ind_07_bin"] + (-(data["ps_ind_03"])))/2.0)))))
    v["128"] = 0.020000*np.tanh((((data["missing"] - data["ps_car_11"]) * (data["loo_ps_car_01_cat"] - data["loo_ps_car_06_cat"])) - data["loo_ps_car_11_cat"]))
    v["129"] = 0.020000*np.tanh((data["ps_reg_01"] * (-((np.tanh((data["ps_reg_03"] + data["loo_ps_ind_05_cat"])) + data["loo_ps_ind_18_bin"])))))
    v["130"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * ((data["missing"] + data["ps_car_11"]) + data["ps_ind_15"])))
    v["131"] = 0.019996*np.tanh((((data["loo_ps_ind_02_cat"] - data["loo_ps_car_04_cat"]) + ((data["loo_ps_ind_07_bin"] * data["ps_car_13"]) * data["ps_reg_03"]))/2.0))
    v["132"] = 0.020000*np.tanh((data["ps_reg_01"] - (0.791667 - (((data["loo_ps_car_11_cat"] * data["loo_ps_car_11_cat"]) + data["loo_ps_ind_02_cat"])/2.0))))
    v["133"] = 0.020000*np.tanh(((data["ps_ind_14"] + ((data["loo_ps_ind_04_cat"] + data["ps_ind_15"])/2.0)) * ((data["ps_ind_15"] + data["loo_ps_ind_04_cat"])/2.0)))
    v["134"] = 0.020000*np.tanh(((0.791667 - data["ps_ind_01"]) * (((data["ps_ind_03"] + data["loo_ps_ind_04_cat"])/2.0) * data["loo_ps_ind_05_cat"])))
    v["135"] = 0.019992*np.tanh(((-2.0 + (data["ps_car_12"] * ((data["ps_car_12"] - 1.089890) + data["loo_ps_ind_04_cat"])))/2.0))
    v["136"] = 0.019977*np.tanh(((data["loo_ps_car_09_cat"] * (data["loo_ps_car_01_cat"] * (data["loo_ps_ind_02_cat"] + data["loo_ps_ind_17_bin"]))) - data["loo_ps_ind_02_cat"]))
    v["137"] = 0.019996*np.tanh(((-1.0 + ((data["ps_reg_03"] * data["ps_reg_03"]) * data["loo_ps_ind_05_cat"])) * data["loo_ps_ind_05_cat"]))
    v["138"] = 0.020000*np.tanh((((data["ps_car_11"] * (data["ps_ind_14"] - data["ps_car_12"])) + data["loo_ps_car_09_cat"])/2.0))
    v["139"] = 0.019996*np.tanh((data["loo_ps_car_07_cat"] * ((((data["missing"] * data["missing"]) + data["ps_ind_15"])/2.0) + data["loo_ps_ind_17_bin"])))
    v["140"] = 0.020000*np.tanh((data["ps_reg_01"] * ((data["ps_reg_01"] - data["ps_reg_03"]) - data["loo_ps_ind_17_bin"])))
    v["141"] = 0.019996*np.tanh(((data["loo_ps_ind_04_cat"] - ((data["ps_car_13"] + (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_05_cat"]))/2.0)) * data["loo_ps_car_07_cat"]))
    v["142"] = 0.020000*np.tanh(((data["ps_ind_15"] + (((data["ps_reg_03"] * data["ps_reg_03"]) - 0.432099) + data["loo_ps_ind_16_bin"]))/2.0))
    v["143"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_08_bin"] - (data["loo_ps_car_04_cat"] + (data["loo_ps_ind_08_bin"] * data["ps_car_13"]))))/2.0))
    v["144"] = 0.019996*np.tanh(((((-1.0 - data["ps_ind_03"]) - data["ps_ind_03"]) * data["loo_ps_ind_02_cat"]) - 0.432099))
    v["145"] = 0.020000*np.tanh((((data["loo_ps_ind_02_cat"] * (data["loo_ps_ind_02_cat"] - 0.788462)) * data["loo_ps_ind_02_cat"]) - data["loo_ps_car_10_cat"]))
    v["146"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] * ((data["loo_ps_ind_04_cat"] + data["ps_ind_03"]) * (0.791667 - data["ps_ind_03"]))))
    v["147"] = 0.020000*np.tanh((np.tanh(data["loo_ps_ind_05_cat"]) * (((-(data["loo_ps_ind_18_bin"])) + (data["ps_car_14"] + data["missing"]))/2.0)))
    v["148"] = 0.020000*np.tanh((data["ps_reg_03"] * (((data["loo_ps_ind_09_bin"] - data["loo_ps_ind_18_bin"]) - data["ps_car_15"]) - data["loo_ps_ind_02_cat"])))
    v["149"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] + (data["ps_ind_15"] * (-((data["loo_ps_ind_17_bin"] - data["ps_ind_14"])))))/2.0))
    v["150"] = 0.020000*np.tanh((((data["loo_ps_ind_04_cat"] * data["loo_ps_car_03_cat"]) + (-(((data["loo_ps_car_01_cat"] + data["loo_ps_car_10_cat"])/2.0))))/2.0))
    v["151"] = 0.020000*np.tanh(((0.666667 - data["ps_ind_03"]) * ((data["loo_ps_car_07_cat"] + (data["ps_reg_01"] + data["ps_ind_15"]))/2.0)))
    v["152"] = 0.020000*np.tanh(((data["loo_ps_car_03_cat"] * data["loo_ps_ind_05_cat"]) * ((data["loo_ps_car_06_cat"] + ((-1.0 + data["loo_ps_ind_04_cat"])/2.0))/2.0)))
    v["153"] = 0.020000*np.tanh(((data["ps_ind_03"] * data["ps_ind_03"]) * (-2.0 + (data["ps_ind_03"] * data["ps_ind_03"]))))
    v["154"] = 0.020000*np.tanh(((data["loo_ps_ind_13_bin"] - data["loo_ps_ind_11_bin"]) * ((data["loo_ps_ind_02_cat"] + (7.0)) * 29.500000)))
    v["155"] = 0.019977*np.tanh((((data["loo_ps_ind_02_cat"] + data["loo_ps_car_08_cat"]) * (data["loo_ps_ind_02_cat"] * 2.0)) - 1.089890))
    v["156"] = 0.020000*np.tanh(((data["ps_car_12"] + (data["loo_ps_ind_16_bin"] * ((-(data["ps_reg_01"])) * data["loo_ps_car_05_cat"])))/2.0))
    v["157"] = 0.020000*np.tanh((((data["loo_ps_ind_04_cat"] - data["loo_ps_ind_05_cat"]) + (data["loo_ps_ind_05_cat"] * data["ps_ind_15"]))/2.0))
    v["158"] = 0.020000*np.tanh(((data["loo_ps_car_01_cat"] * (((data["loo_ps_car_02_cat"] + data["loo_ps_ind_17_bin"])/2.0) * data["loo_ps_ind_02_cat"])) * data["loo_ps_car_07_cat"]))
    v["159"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_16_bin"]) + -3.0) * data["loo_ps_ind_04_cat"])))
    v["160"] = 0.020000*np.tanh((data["loo_ps_car_05_cat"] * ((data["loo_ps_ind_02_cat"] + (-((data["loo_ps_car_06_cat"] * data["loo_ps_car_02_cat"]))))/2.0)))
    v["161"] = 0.019961*np.tanh((-((data["loo_ps_car_03_cat"] * (0.753247 - (data["ps_ind_01"] * data["ps_ind_01"]))))))
    v["162"] = 0.020000*np.tanh(((data["ps_car_13"] + data["loo_ps_ind_17_bin"]) * ((data["ps_ind_03"] + (data["ps_ind_03"] + data["loo_ps_car_07_cat"]))/2.0)))
    v["163"] = 0.019988*np.tanh(((data["ps_ind_01"] + ((data["loo_ps_ind_09_bin"] + ((data["ps_ind_01"] + data["missing"])/2.0))/2.0)) * data["missing"]))
    v["164"] = 0.019988*np.tanh((data["ps_reg_03"] * (-(((((data["loo_ps_ind_02_cat"] + (-(data["ps_reg_03"])))/2.0) + data["ps_reg_01"])/2.0)))))
    v["165"] = 0.020000*np.tanh(((1.480000 + (data["loo_ps_ind_11_bin"] * 29.500000)) * 29.500000))
    v["166"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] - ((2.0 + data["ps_car_11"]) * (2.352940 * 2.352940))))
    v["167"] = 0.020000*np.tanh(((data["ps_car_11"] + (((-(data["loo_ps_car_04_cat"])) + ((data["loo_ps_ind_04_cat"] + data["loo_ps_car_02_cat"])/2.0))/2.0))/2.0))
    v["168"] = 0.019996*np.tanh((((data["loo_ps_car_09_cat"] * (-(data["ps_reg_01"]))) + (data["ps_reg_01"] - data["ps_ind_03"]))/2.0))
    v["169"] = 0.019996*np.tanh((data["loo_ps_car_09_cat"] * ((data["loo_ps_car_09_cat"] * data["loo_ps_ind_17_bin"]) - (0.791667 - data["ps_reg_02"]))))
    v["170"] = 0.019992*np.tanh(((data["ps_car_15"] * (data["loo_ps_ind_06_bin"] - data["ps_ind_03"])) * data["loo_ps_car_04_cat"]))
    v["171"] = 0.020000*np.tanh(((-(data["ps_reg_03"])) * (data["missing"] + ((data["loo_ps_car_03_cat"] + data["ps_car_14"])/2.0))))
    v["172"] = 0.019996*np.tanh((data["ps_ind_01"] * ((data["loo_ps_ind_05_cat"] * ((data["loo_ps_ind_02_cat"] + data["ps_ind_01"])/2.0)) - data["loo_ps_ind_04_cat"])))
    v["173"] = 0.019953*np.tanh(((data["ps_reg_01"] * (-(((data["loo_ps_ind_06_bin"] + np.tanh(data["loo_ps_ind_17_bin"]))/2.0)))) - data["loo_ps_car_10_cat"]))
    v["174"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (((data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"]) - 0.887097) + data["ps_ind_01"])))
    v["175"] = 0.020000*np.tanh((((data["ps_reg_01"] + (data["loo_ps_car_01_cat"] * 0.117647))/2.0) * (data["loo_ps_car_01_cat"] * data["loo_ps_car_01_cat"])))
    v["176"] = 0.019988*np.tanh((((data["loo_ps_ind_04_cat"] * np.tanh(data["loo_ps_ind_17_bin"])) + np.tanh((-3.0 + data["ps_car_13"])))/2.0))
    v["177"] = 0.020000*np.tanh((1.089890 + ((11.83836174011230469) * ((0.666667 * 1.089890) - data["ps_car_15"]))))
    v["178"] = 0.019996*np.tanh(((-1.0 + (data["loo_ps_ind_17_bin"] * ((data["loo_ps_ind_09_bin"] + ((data["loo_ps_ind_04_cat"] + -3.0)/2.0))/2.0)))/2.0))
    v["179"] = 0.020000*np.tanh((data["loo_ps_car_03_cat"] * (data["loo_ps_car_04_cat"] * (data["loo_ps_ind_04_cat"] - np.tanh(data["ps_car_15"])))))
    v["180"] = 0.019887*np.tanh((data["loo_ps_car_08_cat"] + (data["ps_ind_03"] * (data["loo_ps_car_02_cat"] * data["ps_ind_01"]))))
    v["181"] = 0.020000*np.tanh(((((data["ps_reg_01"] + (-(data["ps_car_13"])))/2.0) * data["ps_ind_15"]) * data["loo_ps_car_01_cat"]))
    v["182"] = 0.020000*np.tanh((data["ps_reg_01"] * (data["loo_ps_ind_07_bin"] * (((data["loo_ps_ind_02_cat"] + data["loo_ps_car_08_cat"])/2.0) - data["ps_reg_03"]))))
    v["183"] = 0.020000*np.tanh((((data["ps_ind_14"] + (data["loo_ps_ind_04_cat"] * data["loo_ps_ind_17_bin"]))/2.0) * (data["loo_ps_car_01_cat"] * data["loo_ps_ind_04_cat"])))
    v["184"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] * ((data["loo_ps_ind_02_cat"] + (data["ps_ind_15"] * (-(data["ps_car_15"]))))/2.0)))
    v["185"] = 0.019973*np.tanh((data["loo_ps_ind_17_bin"] * ((data["loo_ps_car_01_cat"] - data["ps_ind_14"]) * (data["ps_car_11"] + data["loo_ps_car_09_cat"]))))
    v["186"] = 0.019918*np.tanh((data["loo_ps_ind_02_cat"] * (((data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"]) - 0.666667) + data["loo_ps_ind_18_bin"])))
    v["187"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (data["loo_ps_ind_04_cat"] * (data["ps_car_13"] + (data["loo_ps_car_04_cat"] + data["ps_car_12"])))))
    v["188"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] * (data["loo_ps_ind_05_cat"] * (data["ps_car_13"] + data["ps_car_13"]))))
    v["189"] = 0.019992*np.tanh((data["ps_car_13"] * (data["loo_ps_car_01_cat"] * (data["ps_car_13"] - 2.0))))
    v["190"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] * (data["loo_ps_ind_05_cat"] * (-(data["ps_reg_02"])))))
    v["191"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] * (((data["loo_ps_ind_02_cat"] * 0.432099) * data["loo_ps_car_07_cat"]) - data["loo_ps_car_02_cat"])))
    v["192"] = 0.020000*np.tanh((data["ps_reg_01"] * (data["loo_ps_car_07_cat"] * (-((data["loo_ps_ind_02_cat"] + data["loo_ps_ind_18_bin"]))))))
    v["193"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] * (data["loo_ps_car_08_cat"] * (data["ps_reg_03"] - ((data["ps_ind_03"] + data["ps_ind_03"])/2.0)))))
    v["194"] = 0.019996*np.tanh(((data["ps_ind_03"] - ((0.600000 + data["loo_ps_ind_08_bin"])/2.0)) * ((data["ps_ind_14"] + data["loo_ps_ind_08_bin"])/2.0)))
    v["195"] = 0.019992*np.tanh(((data["loo_ps_car_11_cat"] * data["loo_ps_car_04_cat"]) * (((data["ps_ind_01"] + data["ps_ind_01"]) + data["loo_ps_ind_06_bin"])/2.0)))
    v["196"] = 0.019996*np.tanh(((data["loo_ps_car_04_cat"] - ((data["ps_ind_01"] + data["loo_ps_ind_02_cat"])/2.0)) * (data["loo_ps_ind_02_cat"] * data["ps_ind_03"])))
    v["197"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * (data["ps_car_11"] * ((data["ps_reg_02"] - data["loo_ps_car_01_cat"]) - data["loo_ps_car_03_cat"]))))
    v["198"] = 0.020000*np.tanh((data["ps_reg_03"] * ((((data["loo_ps_car_04_cat"] * data["loo_ps_car_08_cat"]) + data["loo_ps_ind_02_cat"])/2.0) * data["loo_ps_car_08_cat"])))
    v["199"] = 0.020000*np.tanh((((data["ps_car_11"] + data["loo_ps_car_04_cat"])/2.0) * ((data["loo_ps_car_04_cat"] * data["loo_ps_ind_02_cat"]) - data["loo_ps_car_04_cat"])))
    v["200"] = 0.019992*np.tanh((((data["ps_car_13"] * data["ps_ind_14"]) * data["ps_ind_15"]) * (data["ps_car_13"] * data["ps_car_13"])))
    v["201"] = 0.019644*np.tanh((data["ps_ind_15"] * (data["ps_reg_03"] + ((data["ps_ind_15"] * data["loo_ps_car_08_cat"]) + data["ps_reg_03"]))))
    v["202"] = 0.020000*np.tanh(((-((data["ps_reg_02"] * ((data["loo_ps_car_08_cat"] + data["ps_ind_15"])/2.0)))) * data["ps_ind_03"]))
    v["203"] = 0.020000*np.tanh((0.666667 - (data["ps_car_15"] * (((data["loo_ps_car_08_cat"] + data["loo_ps_car_10_cat"]) + data["loo_ps_car_03_cat"])/2.0))))
    v["204"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] * np.tanh(data["loo_ps_car_02_cat"])) + (data["loo_ps_car_09_cat"] * data["loo_ps_car_11_cat"])))
    v["205"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] * (data["loo_ps_ind_08_bin"] - ((data["loo_ps_ind_18_bin"] * data["ps_reg_01"]) * data["ps_reg_01"]))))
    v["206"] = 0.019980*np.tanh((data["loo_ps_ind_04_cat"] * ((data["loo_ps_car_01_cat"] - (data["loo_ps_ind_17_bin"] * data["loo_ps_car_01_cat"])) * data["ps_reg_02"])))
    v["207"] = 0.020000*np.tanh((((data["loo_ps_ind_08_bin"] * (data["ps_ind_15"] + data["loo_ps_car_01_cat"])) - data["loo_ps_car_10_cat"]) - data["loo_ps_car_10_cat"]))
    v["208"] = 0.020000*np.tanh((0.148148 * (data["loo_ps_ind_02_cat"] * (-((np.tanh(data["loo_ps_ind_18_bin"]) * data["loo_ps_ind_16_bin"]))))))
    v["209"] = 0.020000*np.tanh((((data["ps_car_15"] + data["loo_ps_ind_04_cat"])/2.0) * (data["ps_car_12"] * ((data["loo_ps_ind_16_bin"] + data["ps_car_12"])/2.0))))
    v["210"] = 0.020000*np.tanh((data["loo_ps_car_11_cat"] * (data["loo_ps_ind_17_bin"] * ((data["loo_ps_ind_02_cat"] + (-(data["loo_ps_car_07_cat"])))/2.0))))
    v["211"] = 0.020000*np.tanh(((data["ps_ind_03"] * ((data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_02_cat"] * data["ps_ind_03"]))/2.0)) - data["loo_ps_ind_02_cat"]))
    v["212"] = 0.019711*np.tanh(((((data["loo_ps_ind_09_bin"] * data["loo_ps_car_03_cat"]) + (data["loo_ps_ind_09_bin"] * data["loo_ps_car_07_cat"]))/2.0) - 0.232323))
    v["213"] = 0.020000*np.tanh((data["ps_ind_03"] * (0.633333 - ((data["ps_ind_01"] + data["loo_ps_ind_02_cat"]) * data["ps_ind_01"]))))
    v["214"] = 0.020000*np.tanh((data["loo_ps_ind_12_bin"] * (data["loo_ps_ind_11_bin"] + ((data["ps_ind_15"] + data["ps_reg_03"]) + data["ps_reg_03"]))))
    v["215"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * ((data["loo_ps_car_08_cat"] * (-(data["loo_ps_ind_02_cat"]))) - data["loo_ps_car_07_cat"])))
    v["216"] = 0.020000*np.tanh(((((data["loo_ps_ind_17_bin"] + data["loo_ps_car_09_cat"])/2.0) * data["loo_ps_ind_05_cat"]) * data["ps_reg_02"]))
    v["217"] = 0.019984*np.tanh((data["ps_car_14"] * (data["loo_ps_car_07_cat"] * ((data["ps_ind_15"] + data["loo_ps_ind_09_bin"])/2.0))))
    v["218"] = 0.020000*np.tanh((((data["loo_ps_ind_09_bin"] + ((data["loo_ps_ind_02_cat"] + data["ps_reg_02"])/2.0))/2.0) * ((data["missing"] + data["ps_reg_02"])/2.0)))
    v["219"] = 0.019984*np.tanh((data["ps_car_14"] * ((data["loo_ps_car_08_cat"] * (-(data["loo_ps_ind_04_cat"]))) + (-(data["loo_ps_car_07_cat"])))))
    v["220"] = 0.020000*np.tanh((((data["ps_ind_03"] * ((data["loo_ps_ind_02_cat"] * data["ps_ind_03"]) * 2.0)) + data["ps_ind_03"])/2.0))
    v["221"] = 0.019949*np.tanh((data["loo_ps_ind_06_bin"] * (data["loo_ps_ind_16_bin"] * ((data["loo_ps_car_07_cat"] + ((data["ps_reg_02"] + data["loo_ps_ind_08_bin"])/2.0))/2.0))))
    v["222"] = 0.019984*np.tanh(((data["loo_ps_ind_17_bin"] * data["ps_reg_03"]) * (-(((data["loo_ps_ind_12_bin"] + data["loo_ps_car_04_cat"])/2.0)))))
    v["223"] = 0.019848*np.tanh(((data["ps_ind_01"] * (data["loo_ps_ind_18_bin"] * data["loo_ps_car_01_cat"])) - (data["loo_ps_car_10_cat"] * data["ps_ind_03"])))
    v["224"] = 0.019973*np.tanh((((data["ps_ind_03"] * 1.480000) * data["ps_reg_03"]) * ((data["ps_car_12"] + data["loo_ps_ind_17_bin"])/2.0)))
    v["225"] = 0.020000*np.tanh((-((data["loo_ps_ind_17_bin"] * (data["ps_car_15"] - (data["loo_ps_ind_04_cat"] * data["loo_ps_car_10_cat"]))))))
    v["226"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] + data["loo_ps_ind_04_cat"]) * (data["ps_ind_01"] * (-(data["ps_ind_03"])))))
    v["227"] = 0.020000*np.tanh((-((((data["loo_ps_ind_06_bin"] + (data["loo_ps_car_08_cat"] + data["loo_ps_ind_02_cat"]))/2.0) * np.tanh(data["loo_ps_ind_18_bin"])))))
    v["228"] = 0.019988*np.tanh((data["loo_ps_ind_07_bin"] * (data["loo_ps_ind_17_bin"] * (((data["loo_ps_car_08_cat"] + data["ps_reg_02"]) + data["loo_ps_ind_05_cat"])/2.0))))
    v["229"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] * (data["ps_ind_15"] * (data["ps_car_12"] - np.tanh(0.753247)))))
    v["230"] = 0.014245*np.tanh((((data["ps_reg_02"] + data["loo_ps_car_08_cat"])/2.0) * (data["ps_reg_02"] * (data["ps_reg_02"] - 3.0))))
    v["231"] = 0.020000*np.tanh(((0.232323 + (data["loo_ps_car_11_cat"] * (data["ps_ind_01"] * (data["ps_reg_03"] * data["ps_reg_02"]))))/2.0))
    v["232"] = 0.019984*np.tanh((data["loo_ps_ind_08_bin"] * (-(((((data["ps_reg_03"] + data["loo_ps_car_11_cat"])/2.0) + data["loo_ps_ind_12_bin"])/2.0)))))
    v["233"] = 0.019969*np.tanh((-(np.tanh((data["ps_reg_02"] * (data["loo_ps_car_06_cat"] - (data["ps_ind_03"] * data["loo_ps_ind_18_bin"])))))))
    v["234"] = 0.019996*np.tanh((-(((data["ps_car_15"] + ((data["ps_car_14"] * data["loo_ps_car_02_cat"]) - 0.753247))/2.0))))
    v["235"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] * (-(data["loo_ps_ind_17_bin"]))) * (data["loo_ps_car_02_cat"] + data["loo_ps_car_06_cat"])))
    v["236"] = 0.019984*np.tanh((data["loo_ps_car_11_cat"] * (data["loo_ps_car_01_cat"] * (data["ps_reg_01"] * ((0.791667 + data["loo_ps_car_07_cat"])/2.0)))))
    v["237"] = 0.020000*np.tanh((data["loo_ps_car_08_cat"] * (((-((data["loo_ps_car_03_cat"] * data["ps_ind_01"]))) + data["ps_ind_01"])/2.0)))
    v["238"] = 0.019973*np.tanh((data["ps_car_11"] * ((data["loo_ps_car_06_cat"] + (data["ps_ind_03"] * np.tanh((-(data["ps_car_11"])))))/2.0)))
    v["239"] = 0.020000*np.tanh((data["loo_ps_car_01_cat"] * (data["loo_ps_car_06_cat"] * (((-(data["loo_ps_car_07_cat"])) + data["loo_ps_car_05_cat"])/2.0))))
    v["240"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] * (data["ps_ind_01"] * (data["loo_ps_car_09_cat"] - (0.945455 - data["ps_ind_01"])))))
    v["241"] = 0.020000*np.tanh((((data["loo_ps_ind_02_cat"] * (data["ps_ind_15"] - data["loo_ps_ind_02_cat"])) * data["ps_ind_15"]) - data["loo_ps_ind_02_cat"]))
    v["242"] = 0.019687*np.tanh(((-3.0 * ((data["ps_ind_15"] + 1.0) + data["ps_ind_03"])) * data["loo_ps_car_09_cat"]))
    v["243"] = 0.020000*np.tanh((((0.633333 * data["ps_car_14"]) * data["loo_ps_ind_17_bin"]) * (data["loo_ps_ind_04_cat"] - data["ps_car_12"])))
    v["244"] = 0.019930*np.tanh((data["loo_ps_ind_08_bin"] * ((data["ps_ind_03"] + ((data["missing"] + (data["loo_ps_ind_04_cat"] - 1.0))/2.0))/2.0)))
    v["245"] = 0.019293*np.tanh((data["loo_ps_ind_04_cat"] * (np.tanh(data["ps_car_12"]) + (-(data["loo_ps_car_06_cat"])))))
    v["246"] = 0.020000*np.tanh((data["loo_ps_ind_08_bin"] * ((data["loo_ps_ind_08_bin"] * data["loo_ps_ind_08_bin"]) * (data["ps_ind_01"] - data["loo_ps_car_05_cat"]))))
    v["247"] = 0.020000*np.tanh((np.tanh((data["ps_ind_15"] * (data["loo_ps_ind_07_bin"] + data["ps_ind_15"]))) * data["loo_ps_ind_07_bin"]))
    v["248"] = 0.019254*np.tanh(((((data["loo_ps_ind_08_bin"] + (data["ps_car_12"] * data["ps_ind_14"]))/2.0) + (data["loo_ps_ind_08_bin"] * data["loo_ps_car_03_cat"]))/2.0))
    v["249"] = 0.019992*np.tanh(((data["loo_ps_ind_13_bin"] + (-((data["ps_ind_03"] * (data["ps_car_14"] + data["loo_ps_ind_02_cat"])))))/2.0))
    v["250"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] * data["loo_ps_car_08_cat"]) * (data["loo_ps_car_03_cat"] - (data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"]))))
    v["251"] = 0.020000*np.tanh(((np.tanh(data["ps_car_12"]) * data["loo_ps_ind_02_cat"]) - ((data["loo_ps_ind_02_cat"] + np.tanh(data["loo_ps_ind_05_cat"]))/2.0)))
    v["252"] = 0.020000*np.tanh(((data["ps_ind_15"] * data["ps_ind_15"]) * (data["loo_ps_ind_12_bin"] * (data["ps_ind_15"] * data["loo_ps_car_11_cat"]))))
    v["253"] = 0.019801*np.tanh(((data["loo_ps_ind_05_cat"] - 0.232323) * ((data["loo_ps_ind_04_cat"] - data["ps_car_11"]) * data["loo_ps_ind_17_bin"])))
    v["254"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] * (data["ps_ind_14"] * (-((data["loo_ps_car_03_cat"] + data["loo_ps_ind_04_cat"]))))))
    v["255"] = 0.019711*np.tanh((((data["ps_car_11"] + data["ps_ind_01"])/2.0) * (-((data["loo_ps_car_04_cat"] * np.tanh(data["ps_ind_01"]))))))
    v["256"] = 0.019996*np.tanh((data["loo_ps_ind_02_cat"] * ((data["ps_car_15"] * data["loo_ps_ind_17_bin"]) * (data["missing"] - 0.945455))))
    v["257"] = 0.020000*np.tanh((((-((data["missing"] * data["loo_ps_car_07_cat"]))) + (data["loo_ps_car_07_cat"] * (-(data["loo_ps_ind_12_bin"]))))/2.0))
    v["258"] = 0.019844*np.tanh(((data["loo_ps_ind_05_cat"] + data["ps_car_14"]) * (-((data["ps_ind_01"] * data["loo_ps_ind_17_bin"])))))
    v["259"] = 0.020000*np.tanh((data["loo_ps_ind_08_bin"] * (((data["missing"] + np.tanh(data["loo_ps_ind_02_cat"]))/2.0) * data["ps_car_14"])))
    v["260"] = 0.019945*np.tanh(((data["loo_ps_ind_04_cat"] * (data["loo_ps_ind_04_cat"] + (-(data["loo_ps_ind_09_bin"])))) - data["loo_ps_car_10_cat"]))
    v["261"] = 0.020000*np.tanh((((data["ps_ind_03"] + data["loo_ps_ind_12_bin"])/2.0) * (data["ps_ind_03"] + np.tanh((-(data["ps_ind_03"]))))))
    v["262"] = 0.019992*np.tanh((data["loo_ps_car_10_cat"] * (((data["loo_ps_ind_04_cat"] * data["loo_ps_ind_05_cat"]) * data["ps_ind_01"]) - data["ps_ind_01"])))
    v["263"] = 0.020000*np.tanh((data["ps_reg_02"] * ((data["ps_car_13"] - data["loo_ps_car_07_cat"]) * (data["ps_car_12"] - 0.633333))))
    v["264"] = 0.019973*np.tanh((data["ps_ind_14"] * (((8.428570 * (data["loo_ps_ind_07_bin"] * data["ps_car_15"])) + data["loo_ps_ind_13_bin"])/2.0)))
    v["265"] = 0.020000*np.tanh(((data["loo_ps_car_03_cat"] * (data["loo_ps_ind_05_cat"] - data["loo_ps_car_10_cat"])) * np.tanh(data["ps_reg_02"])))
    v["266"] = 0.020000*np.tanh((((data["ps_car_13"] + np.tanh(data["loo_ps_ind_17_bin"]))/2.0) * (((-(data["loo_ps_car_10_cat"])) + data["loo_ps_ind_04_cat"])/2.0)))
    v["267"] = 0.020000*np.tanh(((data["ps_ind_03"] * data["loo_ps_ind_17_bin"]) * ((-1.0 + (data["loo_ps_car_03_cat"] * data["ps_car_13"]))/2.0)))
    v["268"] = 0.019977*np.tanh(((((data["loo_ps_ind_11_bin"] + (data["ps_car_11"] + 1.480000))/2.0) + data["ps_ind_15"]) * data["loo_ps_ind_12_bin"]))
    v["269"] = 0.020000*np.tanh((np.tanh(np.tanh(data["ps_reg_03"])) * (data["ps_car_11"] * (data["ps_ind_14"] + data["ps_ind_15"]))))
    v["270"] = 0.018535*np.tanh((((data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_18_bin"] - (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_18_bin"])))/2.0) * data["loo_ps_ind_09_bin"]))
    v["271"] = 0.020000*np.tanh((data["ps_car_15"] * (data["ps_car_12"] * (np.tanh(data["ps_car_15"]) - data["ps_car_11"]))))
    v["272"] = 0.019957*np.tanh((data["ps_reg_02"] * (data["ps_reg_03"] * (-3.0 + data["ps_reg_02"]))))
    v["273"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - data["loo_ps_ind_12_bin"]) * ((data["ps_car_15"] + ((data["loo_ps_ind_02_cat"] + data["loo_ps_ind_02_cat"])/2.0))/2.0)))
    v["274"] = 0.020000*np.tanh((data["loo_ps_car_10_cat"] * ((data["loo_ps_ind_17_bin"] * data["loo_ps_ind_02_cat"]) - data["loo_ps_ind_16_bin"])))
    v["275"] = 0.020000*np.tanh((((data["ps_ind_03"] + (-((data["loo_ps_ind_09_bin"] * data["ps_ind_15"]))))/2.0) * data["loo_ps_ind_05_cat"]))
    v["276"] = 0.019512*np.tanh(np.tanh((data["ps_ind_03"] * ((data["ps_reg_03"] * data["ps_reg_03"]) - 0.791667))))
    v["277"] = 0.018703*np.tanh((-(((0.633333 * data["loo_ps_ind_05_cat"]) * (data["loo_ps_ind_16_bin"] * data["loo_ps_car_08_cat"])))))
    v["278"] = 0.019988*np.tanh(((data["loo_ps_ind_02_cat"] * data["ps_ind_03"]) * (data["ps_ind_03"] * data["loo_ps_ind_06_bin"])))
    v["279"] = 0.019723*np.tanh((((data["loo_ps_ind_09_bin"] * data["loo_ps_car_04_cat"]) + (data["ps_ind_03"] * data["ps_car_13"])) * data["ps_car_15"]))
    v["280"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] + (data["loo_ps_ind_02_cat"] + data["ps_ind_03"])) * (data["loo_ps_ind_13_bin"] - data["loo_ps_ind_02_cat"])))
    v["281"] = 0.019992*np.tanh((data["loo_ps_ind_04_cat"] * (data["loo_ps_ind_05_cat"] * (data["ps_ind_01"] - ((0.432099 + data["loo_ps_car_06_cat"])/2.0)))))
    v["282"] = 0.020000*np.tanh((data["ps_reg_01"] * (((data["loo_ps_ind_05_cat"] * (-(data["ps_car_12"]))) + data["loo_ps_ind_11_bin"])/2.0)))
    v["283"] = 0.020000*np.tanh((((-(data["ps_ind_03"])) * data["loo_ps_ind_05_cat"]) * np.tanh(np.tanh(data["ps_ind_01"]))))
    v["284"] = 0.018035*np.tanh((data["ps_ind_03"] * np.tanh((data["loo_ps_car_05_cat"] * (data["loo_ps_ind_16_bin"] - data["ps_ind_01"])))))
    v["285"] = 0.017937*np.tanh((data["loo_ps_ind_04_cat"] * ((data["ps_car_11"] + (data["loo_ps_ind_04_cat"] - data["loo_ps_ind_16_bin"])) + data["ps_ind_03"])))
    v["286"] = 0.020000*np.tanh((data["ps_car_13"] * (data["loo_ps_car_03_cat"] * ((data["loo_ps_ind_13_bin"] * data["ps_car_13"]) * data["ps_car_13"]))))
    v["287"] = 0.019140*np.tanh((data["loo_ps_car_03_cat"] * ((data["loo_ps_car_02_cat"] * (data["ps_reg_02"] - data["ps_reg_01"])) * data["loo_ps_ind_04_cat"])))
    v["288"] = 0.019984*np.tanh((data["loo_ps_car_06_cat"] * (data["loo_ps_ind_02_cat"] * (data["loo_ps_ind_02_cat"] * (-(data["loo_ps_ind_16_bin"]))))))
    v["289"] = 0.017492*np.tanh((((data["ps_car_11"] * data["loo_ps_car_06_cat"]) + ((-(data["ps_car_11"])) * data["loo_ps_ind_17_bin"]))/2.0))
    v["290"] = 0.019707*np.tanh((data["loo_ps_ind_04_cat"] * (((data["loo_ps_ind_11_bin"] + (data["missing"] * data["loo_ps_car_07_cat"])) + data["missing"])/2.0)))
    v["291"] = 0.019598*np.tanh(((data["ps_reg_01"] * data["loo_ps_car_04_cat"]) * (data["loo_ps_car_07_cat"] + data["loo_ps_ind_12_bin"])))
    v["292"] = 0.019855*np.tanh(((data["loo_ps_ind_02_cat"] * data["loo_ps_ind_16_bin"]) * (data["loo_ps_ind_07_bin"] * (data["loo_ps_ind_18_bin"] + 0.232323))))
    v["293"] = 0.020000*np.tanh(((data["loo_ps_car_04_cat"] * data["ps_reg_01"]) * np.tanh(((-(data["loo_ps_car_08_cat"])) - data["loo_ps_car_09_cat"]))))
    v["294"] = 0.019992*np.tanh((data["ps_reg_02"] * (data["ps_car_13"] * (data["ps_car_15"] + (data["loo_ps_car_05_cat"] * data["ps_reg_03"])))))
    v["295"] = 0.020000*np.tanh(((data["loo_ps_ind_12_bin"] * (data["loo_ps_car_01_cat"] - data["loo_ps_car_07_cat"])) * data["ps_reg_02"]))
    v["296"] = 0.020000*np.tanh((-((data["loo_ps_car_10_cat"] * ((data["ps_car_15"] + data["loo_ps_car_11_cat"]) + data["ps_ind_14"])))))
    v["297"] = 0.019977*np.tanh((((data["loo_ps_car_07_cat"] + (0.435484 - data["loo_ps_ind_06_bin"]))/2.0) * (data["loo_ps_ind_08_bin"] * data["loo_ps_car_08_cat"])))
    v["298"] = 0.019941*np.tanh((data["ps_car_14"] * (data["loo_ps_ind_04_cat"] + (np.tanh(data["loo_ps_ind_04_cat"]) * (-(data["loo_ps_ind_07_bin"]))))))
    v["299"] = 0.019117*np.tanh((data["loo_ps_ind_06_bin"] * ((data["ps_ind_03"] + data["loo_ps_ind_04_cat"]) * data["loo_ps_ind_17_bin"])))
    v["300"] = 0.020000*np.tanh(((data["loo_ps_ind_16_bin"] * (data["loo_ps_ind_04_cat"] * (data["ps_reg_01"] * data["loo_ps_car_07_cat"]))) * data["loo_ps_ind_18_bin"]))
    v["301"] = 0.019973*np.tanh((-((data["ps_car_11"] * (((data["loo_ps_ind_04_cat"] + data["loo_ps_car_02_cat"])/2.0) * data["loo_ps_ind_05_cat"])))))
    v["302"] = 0.020000*np.tanh((data["ps_car_12"] * ((data["loo_ps_ind_12_bin"] * data["ps_car_12"]) * (data["ps_car_11"] - data["loo_ps_car_08_cat"]))))
    v["303"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] * (data["loo_ps_car_10_cat"] * (data["loo_ps_ind_04_cat"] - (data["ps_ind_03"] * data["loo_ps_ind_17_bin"])))))
    v["304"] = 0.017566*np.tanh(((data["ps_reg_03"] - data["loo_ps_car_03_cat"]) * (data["ps_reg_01"] * data["loo_ps_ind_08_bin"])))
    v["305"] = 0.019973*np.tanh((8.428570 * (0.666667 - (data["loo_ps_car_10_cat"] * (8.428570 - 0.432099)))))
    v["306"] = 0.020000*np.tanh(np.tanh(np.tanh((-3.0 * (((data["ps_reg_01"] - data["loo_ps_car_02_cat"]) + 2.0)/2.0)))))
    v["307"] = 0.020000*np.tanh(((data["ps_ind_01"] - data["ps_reg_01"]) * (data["ps_reg_02"] * data["loo_ps_ind_04_cat"])))
    v["308"] = 0.020000*np.tanh((((data["ps_car_12"] + data["ps_ind_14"])/2.0) * (data["loo_ps_ind_02_cat"] * (data["ps_car_12"] * data["loo_ps_ind_17_bin"]))))
    v["309"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] * data["loo_ps_car_04_cat"]) * ((data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"]) - data["loo_ps_ind_02_cat"])))
    v["310"] = 0.020000*np.tanh((((data["ps_ind_14"] * data["ps_car_12"]) + (data["loo_ps_ind_04_cat"] * (data["loo_ps_ind_04_cat"] + data["ps_ind_14"])))/2.0))
    v["311"] = 0.017878*np.tanh((data["ps_reg_03"] * (-(((data["ps_car_12"] + (data["loo_ps_car_09_cat"] * data["loo_ps_ind_08_bin"]))/2.0)))))
    v["312"] = 0.019973*np.tanh((data["loo_ps_car_01_cat"] * (((data["loo_ps_car_01_cat"] + data["ps_reg_01"])/2.0) * data["ps_car_12"])))
    v["313"] = 0.018691*np.tanh(((data["ps_car_14"] * data["loo_ps_car_11_cat"]) * (-((data["ps_ind_14"] + data["loo_ps_ind_08_bin"])))))
    v["314"] = 0.019566*np.tanh(((1.0 - data["loo_ps_car_11_cat"]) * (((data["missing"] * data["loo_ps_car_11_cat"]) + data["ps_car_14"])/2.0)))
    v["315"] = 0.015312*np.tanh(((data["missing"] + (((data["missing"] - data["loo_ps_ind_02_cat"]) - data["loo_ps_ind_09_bin"]) * data["ps_reg_01"]))/2.0))
    v["316"] = 0.019980*np.tanh((data["loo_ps_car_03_cat"] * ((data["loo_ps_car_11_cat"] * data["loo_ps_ind_04_cat"]) * data["loo_ps_ind_05_cat"])))
    v["317"] = 0.016890*np.tanh((data["loo_ps_car_07_cat"] * ((data["loo_ps_car_11_cat"] - data["ps_car_13"]) - data["loo_ps_ind_06_bin"])))
    v["318"] = 0.020000*np.tanh((data["ps_car_12"] * ((data["ps_reg_02"] * data["loo_ps_car_04_cat"]) * ((data["ps_reg_02"] + data["ps_ind_03"])/2.0))))
    v["319"] = 0.016683*np.tanh(np.tanh((data["loo_ps_ind_16_bin"] * (data["loo_ps_ind_12_bin"] - (data["loo_ps_ind_17_bin"] * data["ps_ind_01"])))))
    v["320"] = 0.018222*np.tanh(((data["loo_ps_ind_13_bin"] * data["ps_car_12"]) - np.tanh((data["loo_ps_ind_02_cat"] + data["ps_car_14"]))))
    v["321"] = 0.020000*np.tanh(((data["ps_ind_14"] * data["ps_car_11"]) * (data["ps_reg_02"] - ((data["ps_car_11"] + data["ps_car_15"])/2.0))))
    v["322"] = 0.019703*np.tanh((((data["ps_ind_01"] + (-(data["loo_ps_car_08_cat"])))/2.0) * (data["loo_ps_ind_17_bin"] * data["ps_ind_01"])))
    v["323"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] * (data["ps_reg_01"] - 0.753247)) * (data["loo_ps_car_02_cat"] * data["loo_ps_car_02_cat"])))
    v["324"] = 0.020000*np.tanh((data["ps_reg_01"] * ((data["loo_ps_car_11_cat"] + ((data["ps_reg_01"] + (-(data["loo_ps_car_04_cat"])))/2.0))/2.0)))
    v["325"] = 0.019898*np.tanh((-(((((data["loo_ps_ind_17_bin"] + (-(data["ps_car_12"])))/2.0) + np.tanh(data["ps_car_12"]))/2.0))))
    v["326"] = 0.019980*np.tanh(((data["loo_ps_ind_12_bin"] + (data["loo_ps_car_04_cat"] * (data["loo_ps_car_09_cat"] + 2.352940))) * data["loo_ps_ind_13_bin"]))
    v["327"] = 0.019996*np.tanh(((((data["loo_ps_ind_08_bin"] * data["loo_ps_car_09_cat"]) + (data["ps_reg_03"] * data["loo_ps_ind_08_bin"]))/2.0) * data["ps_car_15"]))
    v["328"] = 0.020000*np.tanh((data["ps_reg_03"] * (((np.tanh(data["ps_car_11"]) + data["loo_ps_car_09_cat"])/2.0) * np.tanh(data["loo_ps_ind_07_bin"]))))
    v["329"] = 0.019996*np.tanh((((((data["loo_ps_ind_02_cat"] + data["loo_ps_car_10_cat"])/2.0) + data["ps_car_14"])/2.0) * (data["loo_ps_car_08_cat"] * data["ps_reg_02"])))
    v["330"] = 0.020000*np.tanh(((data["loo_ps_car_10_cat"] * data["ps_reg_02"]) * (data["loo_ps_ind_18_bin"] - (data["ps_ind_14"] * data["loo_ps_ind_02_cat"]))))
    v["331"] = 0.020000*np.tanh((((data["ps_ind_14"] + np.tanh(data["ps_ind_01"]))/2.0) * (data["loo_ps_ind_17_bin"] * data["loo_ps_car_06_cat"])))
    v["332"] = 0.020000*np.tanh((((data["ps_ind_15"] + data["loo_ps_ind_05_cat"])/2.0) * np.tanh((data["loo_ps_ind_17_bin"] * (-(data["loo_ps_car_06_cat"]))))))
    v["333"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] * (data["ps_car_15"] * np.tanh((data["ps_ind_03"] + data["loo_ps_car_11_cat"])))))
    v["334"] = 0.020000*np.tanh((data["ps_car_11"] * (((data["loo_ps_ind_11_bin"] + data["loo_ps_ind_10_bin"])/2.0) - (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_04_cat"]))))
    v["335"] = 0.019996*np.tanh((data["ps_car_11"] * ((data["loo_ps_car_10_cat"] + ((data["ps_reg_02"] * data["ps_car_11"]) * data["loo_ps_ind_18_bin"]))/2.0)))
    v["336"] = 0.020000*np.tanh(((data["ps_ind_03"] * data["loo_ps_ind_02_cat"]) * (data["ps_ind_03"] - data["ps_ind_15"])))
    v["337"] = 0.019996*np.tanh((((data["loo_ps_car_07_cat"] * (data["loo_ps_ind_13_bin"] - data["loo_ps_ind_11_bin"])) * data["loo_ps_car_07_cat"]) * data["loo_ps_car_07_cat"]))
    v["338"] = 0.019219*np.tanh(((data["loo_ps_ind_02_cat"] * ((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_18_bin"]) - data["loo_ps_ind_02_cat"])) - data["loo_ps_ind_02_cat"]))
    v["339"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] * (np.tanh(0.232323) - (np.tanh(data["loo_ps_car_01_cat"]) + data["loo_ps_ind_12_bin"]))))
    v["340"] = 0.019047*np.tanh(((data["ps_ind_01"] * (data["ps_reg_01"] * data["loo_ps_car_06_cat"])) - (data["ps_ind_14"] * data["ps_reg_01"])))
    v["341"] = 0.019996*np.tanh((data["loo_ps_car_10_cat"] * ((np.tanh(np.tanh(data["ps_car_12"])) - data["ps_car_13"]) - data["loo_ps_ind_12_bin"])))
    v["342"] = 0.019996*np.tanh((data["ps_car_14"] * (data["loo_ps_car_05_cat"] * (((data["loo_ps_car_05_cat"] + data["ps_car_15"]) + data["ps_reg_01"])/2.0))))
    v["343"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (np.tanh(((-(data["loo_ps_car_07_cat"])) + data["ps_reg_02"])) + data["loo_ps_ind_10_bin"])))
    v["344"] = 0.019902*np.tanh((data["ps_car_14"] * ((np.tanh(data["loo_ps_ind_05_cat"]) * data["ps_ind_15"]) - np.tanh(data["ps_reg_03"]))))
    v["345"] = 0.019937*np.tanh(((data["ps_car_11"] * (data["ps_car_12"] * data["loo_ps_ind_09_bin"])) * (data["loo_ps_ind_11_bin"] - data["loo_ps_ind_18_bin"])))
    v["346"] = 0.019984*np.tanh(((data["loo_ps_car_06_cat"] * (0.232323 - (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_05_cat"]))) * data["loo_ps_ind_04_cat"]))
    v["347"] = 0.020000*np.tanh(((data["loo_ps_ind_11_bin"] * data["ps_ind_03"]) * ((data["loo_ps_ind_05_cat"] * data["ps_ind_03"]) * data["ps_ind_03"])))
    v["348"] = 0.019934*np.tanh((data["loo_ps_ind_16_bin"] * (data["loo_ps_ind_12_bin"] * (data["ps_car_13"] * (data["loo_ps_car_10_cat"] + data["loo_ps_car_03_cat"])))))
    v["349"] = 0.020000*np.tanh(((((data["ps_car_15"] + data["ps_ind_14"])/2.0) * data["loo_ps_ind_06_bin"]) * data["ps_reg_02"]))
    v["350"] = 0.019769*np.tanh(((((data["ps_ind_03"] - data["ps_reg_02"]) * 0.117647) - data["loo_ps_car_10_cat"]) * data["ps_ind_03"]))
    v["351"] = 0.019992*np.tanh((-(((0.232323 + ((data["loo_ps_car_03_cat"] * data["ps_ind_03"]) * data["loo_ps_car_08_cat"]))/2.0))))
    v["352"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((data["ps_ind_03"] * (0.232323 - data["loo_ps_car_11_cat"])) * data["ps_ind_15"])))
    v["353"] = 0.019980*np.tanh((np.tanh(data["ps_ind_03"]) * ((data["ps_car_11"] + (data["ps_reg_03"] + np.tanh(data["missing"])))/2.0)))
    v["354"] = 0.019988*np.tanh((data["ps_car_14"] * (((data["loo_ps_ind_12_bin"] + data["ps_car_12"])/2.0) * data["ps_reg_01"])))
    v["355"] = 0.019957*np.tanh(np.tanh((data["loo_ps_car_07_cat"] * (0.633333 - (data["ps_reg_02"] * data["ps_reg_02"])))))
    v["356"] = 0.019977*np.tanh((data["loo_ps_car_09_cat"] * (data["loo_ps_car_09_cat"] * (data["loo_ps_ind_16_bin"] * (data["ps_reg_01"] * data["ps_ind_15"])))))
    v["357"] = 0.019887*np.tanh(((-((((data["ps_car_14"] * data["loo_ps_ind_12_bin"]) + data["ps_reg_03"])/2.0))) * np.tanh(data["loo_ps_car_06_cat"])))
    v["358"] = 0.019965*np.tanh((data["loo_ps_ind_05_cat"] * (data["ps_reg_02"] * ((np.tanh((-(data["ps_ind_15"]))) + data["loo_ps_car_09_cat"])/2.0))))
    v["359"] = 0.020000*np.tanh((data["loo_ps_ind_12_bin"] * ((data["loo_ps_ind_11_bin"] + (data["loo_ps_car_11_cat"] + data["ps_ind_15"])) - data["loo_ps_ind_02_cat"])))
    v["360"] = 0.020000*np.tanh((((data["ps_reg_01"] + data["ps_car_13"])/2.0) * (data["ps_ind_01"] * data["loo_ps_ind_05_cat"])))
    v["361"] = 0.020000*np.tanh((data["loo_ps_ind_08_bin"] * (-((data["loo_ps_ind_05_cat"] * (data["ps_car_11"] - (-(data["ps_ind_03"]))))))))
    v["362"] = 0.018640*np.tanh(((((-(((data["ps_car_15"] + data["ps_car_11"])/2.0))) + data["loo_ps_car_05_cat"])/2.0) * data["loo_ps_ind_06_bin"]))
    v["363"] = 0.019480*np.tanh((((np.tanh(data["ps_reg_02"]) * ((data["loo_ps_car_07_cat"] + data["loo_ps_ind_11_bin"])/2.0)) + np.tanh(data["loo_ps_car_07_cat"]))/2.0))
    v["364"] = 0.020000*np.tanh((data["ps_ind_14"] * (-((0.753247 - ((data["ps_ind_03"] + data["ps_ind_03"])/2.0))))))
    v["365"] = 0.020000*np.tanh((((data["ps_reg_02"] * (-(data["loo_ps_ind_05_cat"]))) * data["ps_car_15"]) * data["loo_ps_ind_05_cat"]))
    v["366"] = 0.019992*np.tanh((data["ps_ind_15"] * (data["loo_ps_ind_02_cat"] * ((data["loo_ps_ind_02_cat"] * data["ps_reg_01"]) - data["ps_ind_03"]))))
    v["367"] = 0.020000*np.tanh((data["ps_car_15"] * ((0.232323 + ((-(data["ps_ind_15"])) * data["ps_car_15"]))/2.0)))
    v["368"] = 0.019992*np.tanh((((data["loo_ps_car_02_cat"] + data["loo_ps_car_02_cat"]) - data["ps_car_11"]) * (data["loo_ps_ind_02_cat"] * data["loo_ps_ind_04_cat"])))
    v["369"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (data["ps_ind_15"] * ((data["ps_ind_15"] + (data["ps_reg_01"] * data["ps_car_13"]))/2.0))))
    v["370"] = 0.019840*np.tanh((data["ps_car_12"] * ((data["ps_car_11"] + data["loo_ps_ind_12_bin"]) * (-(np.tanh(data["ps_ind_01"]))))))
    v["371"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] * (data["loo_ps_ind_12_bin"] * (data["ps_car_12"] + (data["ps_ind_03"] * data["loo_ps_ind_04_cat"])))))
    v["372"] = 0.017296*np.tanh((data["ps_car_11"] * ((data["ps_reg_03"] + np.tanh((data["loo_ps_car_04_cat"] - data["ps_car_11"])))/2.0)))
    v["373"] = 0.019992*np.tanh((((data["loo_ps_ind_04_cat"] * data["ps_car_11"]) + np.tanh((-((data["ps_car_11"] * data["loo_ps_ind_05_cat"])))))/2.0))
    v["374"] = 0.017175*np.tanh((data["loo_ps_car_09_cat"] * np.tanh(((data["loo_ps_ind_16_bin"] + data["ps_reg_02"]) + data["loo_ps_ind_16_bin"]))))
    v["375"] = 0.020000*np.tanh((np.tanh((0.618557 - data["loo_ps_ind_06_bin"])) * (data["loo_ps_car_08_cat"] * data["loo_ps_ind_02_cat"])))
    v["376"] = 0.019992*np.tanh((np.tanh(data["loo_ps_ind_05_cat"]) * (((-(data["loo_ps_car_02_cat"])) + (data["loo_ps_car_02_cat"] * data["ps_reg_03"]))/2.0)))
    v["377"] = 0.019965*np.tanh(((-(data["loo_ps_car_04_cat"])) * (((data["ps_ind_03"] * data["ps_ind_03"]) + (-(data["ps_car_15"])))/2.0)))
    v["378"] = 0.019777*np.tanh((data["loo_ps_ind_17_bin"] * (data["loo_ps_ind_17_bin"] * (((data["loo_ps_ind_08_bin"] + data["loo_ps_ind_13_bin"])/2.0) * data["loo_ps_car_04_cat"]))))
    v["379"] = 0.019996*np.tanh((data["loo_ps_car_02_cat"] * (((data["loo_ps_car_01_cat"] + data["ps_car_11"])/2.0) * ((data["loo_ps_car_01_cat"] + data["ps_ind_15"])/2.0))))
    v["380"] = 0.019988*np.tanh((data["loo_ps_ind_02_cat"] * (data["ps_car_15"] * (data["ps_car_11"] + (data["ps_ind_15"] * data["ps_reg_01"])))))
    v["381"] = 0.020000*np.tanh(((0.0 + (data["loo_ps_ind_04_cat"] * (data["ps_reg_03"] * (data["loo_ps_car_04_cat"] * data["loo_ps_car_08_cat"]))))/2.0))
    v["382"] = 0.019953*np.tanh(((data["loo_ps_car_01_cat"] * data["loo_ps_car_01_cat"]) * ((data["ps_reg_01"] + (-(data["loo_ps_car_10_cat"])))/2.0)))
    v["383"] = 0.020000*np.tanh((-((data["ps_reg_03"] * (data["ps_ind_14"] * (data["loo_ps_car_04_cat"] * data["ps_reg_03"]))))))
    v["384"] = 0.019078*np.tanh(np.tanh(((data["loo_ps_ind_06_bin"] * data["loo_ps_ind_05_cat"]) * ((-(0.432099)) - data["ps_reg_01"]))))
    v["385"] = 0.019934*np.tanh((data["loo_ps_car_03_cat"] * ((-(data["ps_car_13"])) * (data["loo_ps_car_10_cat"] * data["ps_car_13"]))))
    v["386"] = 0.019945*np.tanh(np.tanh(((data["ps_ind_03"] * ((-(data["loo_ps_ind_04_cat"])) - data["missing"])) * data["loo_ps_ind_07_bin"])))
    v["387"] = 0.019680*np.tanh((data["loo_ps_car_01_cat"] * (((data["loo_ps_car_01_cat"] + (-(data["loo_ps_car_09_cat"])))/2.0) * np.tanh(data["loo_ps_ind_16_bin"]))))
    v["388"] = 0.020000*np.tanh((data["loo_ps_ind_12_bin"] * ((-(data["ps_ind_01"])) * (data["loo_ps_ind_08_bin"] * data["loo_ps_ind_17_bin"]))))
    v["389"] = 0.020000*np.tanh((((data["ps_ind_03"] * 0.232323) + np.tanh((data["ps_reg_03"] * np.tanh(data["loo_ps_car_04_cat"]))))/2.0))
    v["390"] = 0.019988*np.tanh((data["loo_ps_car_08_cat"] * ((data["loo_ps_car_01_cat"] + (data["loo_ps_car_09_cat"] * (data["loo_ps_ind_04_cat"] * data["loo_ps_car_06_cat"])))/2.0)))
    v["391"] = 0.019988*np.tanh(((data["loo_ps_ind_11_bin"] * data["ps_reg_03"]) - ((data["loo_ps_ind_11_bin"] + ((data["loo_ps_ind_16_bin"] + data["loo_ps_ind_10_bin"])/2.0))/2.0)))
    v["392"] = 0.020000*np.tanh(((data["loo_ps_ind_11_bin"] - data["loo_ps_car_10_cat"]) * (data["loo_ps_car_08_cat"] + data["ps_car_13"])))
    v["393"] = 0.019992*np.tanh(((data["loo_ps_ind_05_cat"] * ((data["ps_ind_14"] + (data["ps_car_14"] * data["ps_reg_03"]))/2.0)) * data["loo_ps_car_01_cat"]))
    v["394"] = 0.019238*np.tanh(((np.tanh((-(data["missing"]))) * data["loo_ps_ind_07_bin"]) * (-(data["ps_reg_01"]))))
    v["395"] = 0.019906*np.tanh((-((np.tanh((data["ps_ind_14"] * data["loo_ps_ind_17_bin"])) * (data["loo_ps_ind_07_bin"] + data["loo_ps_ind_07_bin"])))))
    v["396"] = 0.019789*np.tanh(np.tanh((-((data["loo_ps_car_03_cat"] * (data["ps_ind_01"] * (2.0 - data["ps_ind_01"])))))))
    v["397"] = 0.019949*np.tanh(((data["ps_car_13"] * (0.148148 * (-(np.tanh(data["ps_ind_15"]))))) * data["ps_car_13"]))
    v["398"] = 0.019137*np.tanh((data["loo_ps_ind_02_cat"] * (((-(data["ps_ind_03"])) * (-(data["ps_ind_03"]))) - 1.0)))
    v["399"] = 0.019895*np.tanh(np.tanh(np.tanh(((data["loo_ps_car_05_cat"] * data["ps_ind_01"]) + ((data["ps_ind_03"] + data["loo_ps_ind_18_bin"])/2.0)))))
    v["400"] = 0.015581*np.tanh((-((0.232323 * (data["loo_ps_ind_07_bin"] * (data["loo_ps_ind_16_bin"] + data["loo_ps_car_02_cat"]))))))
    return Outputs(v.sum(axis=1))


def GPII(data):
    v = pd.DataFrame()
    v["0"] = -3.274750
    v["1"] = 0.020000*np.tanh((3.642860 * ((data["loo_ps_car_01_cat"] + data["loo_ps_ind_16_bin"]) + (data["ps_reg_03"] + data["loo_ps_ind_06_bin"]))))
    v["2"] = 0.020000*np.tanh((19.500000 * (((data["ps_car_15"] + data["loo_ps_ind_06_bin"]) + data["loo_ps_car_06_cat"]) + data["loo_ps_car_07_cat"])))
    v["3"] = 0.019996*np.tanh((19.500000 * ((data["ps_car_13"] + (data["loo_ps_car_11_cat"] + data["loo_ps_ind_05_cat"])) + data["ps_reg_02"])))
    v["4"] = 0.020000*np.tanh((((data["ps_reg_03"] + (data["loo_ps_ind_05_cat"] + data["loo_ps_car_07_cat"])) + data["ps_car_12"]) * 19.500000))
    v["5"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] + (data["loo_ps_car_04_cat"] - (data["ps_ind_15"] - (data["loo_ps_ind_17_bin"] + data["loo_ps_car_01_cat"])))))
    v["6"] = 0.020000*np.tanh(((11.51410007476806641) * (data["loo_ps_car_11_cat"] + (((data["loo_ps_car_09_cat"] + data["ps_reg_03"])/2.0) + data["loo_ps_ind_06_bin"]))))
    v["7"] = 0.020000*np.tanh((19.500000 * ((data["ps_car_13"] + ((data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_17_bin"] + data["loo_ps_car_01_cat"]))/2.0))/2.0)))
    v["8"] = 0.020000*np.tanh((19.500000 * (((data["ps_reg_02"] + data["loo_ps_ind_09_bin"]) + 0.760563) + data["loo_ps_car_05_cat"])))
    v["9"] = 0.020000*np.tanh((19.500000 * ((data["loo_ps_car_11_cat"] + (((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_06_bin"]) + data["loo_ps_car_03_cat"])/2.0))/2.0)))
    v["10"] = 0.020000*np.tanh((6.846150 * (data["loo_ps_ind_06_bin"] + (data["loo_ps_car_11_cat"] + (data["loo_ps_ind_17_bin"] + data["loo_ps_car_09_cat"])))))
    v["11"] = 0.020000*np.tanh(((9.75283908843994141) * ((data["ps_reg_03"] + (data["loo_ps_car_11_cat"] - data["ps_ind_15"])) + data["loo_ps_ind_04_cat"])))
    v["12"] = 0.020000*np.tanh(((data["ps_reg_02"] + data["loo_ps_car_03_cat"]) + ((data["ps_car_13"] - data["ps_ind_15"]) + data["loo_ps_ind_05_cat"])))
    v["13"] = 0.020000*np.tanh(((8.49180412292480469) * ((data["loo_ps_ind_09_bin"] + data["loo_ps_car_07_cat"]) + (data["loo_ps_car_09_cat"] + data["loo_ps_ind_06_bin"]))))
    v["14"] = 0.020000*np.tanh(((6.0) * ((((data["loo_ps_ind_16_bin"] + data["loo_ps_ind_05_cat"]) + data["ps_reg_01"])/2.0) + data["ps_car_13"])))
    v["15"] = 0.020000*np.tanh((data["loo_ps_car_01_cat"] + (((data["ps_reg_01"] + data["ps_car_12"]) + data["loo_ps_car_03_cat"]) - data["ps_ind_15"])))
    v["16"] = 0.020000*np.tanh(((data["loo_ps_car_08_cat"] + (data["loo_ps_car_04_cat"] + (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"]))) * 3.642860))
    v["17"] = 0.020000*np.tanh((19.500000 * ((0.760563 + (data["ps_reg_02"] + data["ps_car_13"])) + data["ps_reg_03"])))
    v["18"] = 0.020000*np.tanh((data["loo_ps_ind_07_bin"] + ((9.29609489440917969) * (((data["loo_ps_car_03_cat"] + data["ps_reg_03"]) + data["ps_car_15"])/2.0))))
    v["19"] = 0.020000*np.tanh((6.846150 * ((data["loo_ps_ind_05_cat"] + data["ps_car_13"]) + (data["loo_ps_ind_17_bin"] + data["loo_ps_car_07_cat"]))))
    v["20"] = 0.020000*np.tanh((data["loo_ps_car_04_cat"] + (data["ps_car_15"] + (data["ps_reg_03"] + (data["loo_ps_car_03_cat"] - data["ps_ind_15"])))))
    v["21"] = 0.020000*np.tanh((6.846150 * (data["loo_ps_ind_05_cat"] + ((data["ps_car_13"] + (data["ps_ind_03"] + data["loo_ps_ind_17_bin"]))/2.0))))
    v["22"] = 0.020000*np.tanh((19.500000 * ((data["loo_ps_ind_16_bin"] - data["ps_ind_15"]) + (data["ps_reg_03"] + 0.600000))))
    v["23"] = 0.020000*np.tanh((3.0 * (data["loo_ps_car_03_cat"] + (data["loo_ps_car_04_cat"] + (data["loo_ps_ind_05_cat"] + data["loo_ps_car_01_cat"])))))
    v["24"] = 0.020000*np.tanh((6.846150 * (((data["loo_ps_car_01_cat"] + data["ps_ind_03"]) + (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"]))/2.0)))
    v["25"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] + (((data["loo_ps_ind_17_bin"] + data["ps_car_13"]) + data["ps_reg_01"]) + data["loo_ps_ind_09_bin"])))
    v["26"] = 0.020000*np.tanh(((data["loo_ps_ind_09_bin"] + (data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_06_bin"] + data["loo_ps_ind_05_cat"]))) * 19.500000))
    v["27"] = 0.020000*np.tanh(((data["ps_reg_03"] + (data["loo_ps_ind_06_bin"] + (data["loo_ps_car_01_cat"] + 1.871790))) * (13.13490962982177734)))
    v["28"] = 0.020000*np.tanh((6.846150 * (data["loo_ps_ind_05_cat"] + ((data["loo_ps_ind_05_cat"] + (data["ps_car_13"] + data["loo_ps_ind_16_bin"]))/2.0))))
    v["29"] = 0.020000*np.tanh(((data["ps_reg_02"] + (data["loo_ps_ind_17_bin"] + (data["loo_ps_car_11_cat"] + data["ps_car_15"]))) - data["ps_ind_15"]))
    v["30"] = 0.020000*np.tanh(((3.0 * (data["loo_ps_ind_05_cat"] + (data["loo_ps_car_09_cat"] + data["ps_ind_03"]))) + data["ps_car_13"]))
    v["31"] = 0.020000*np.tanh(((8.0) * ((data["loo_ps_ind_09_bin"] + data["loo_ps_ind_05_cat"]) + (data["ps_ind_03"] + data["loo_ps_car_07_cat"]))))
    v["32"] = 0.020000*np.tanh((data["ps_reg_02"] + ((data["ps_car_13"] + data["loo_ps_car_01_cat"]) + ((1.11309552192687988) + data["loo_ps_car_07_cat"]))))
    v["33"] = 0.020000*np.tanh((6.846150 * (((data["loo_ps_ind_05_cat"] - data["ps_ind_15"]) + ((data["loo_ps_ind_16_bin"] + data["loo_ps_car_09_cat"])/2.0))/2.0)))
    v["34"] = 0.020000*np.tanh((19.500000 * ((data["loo_ps_car_07_cat"] + (((data["ps_ind_03"] + data["loo_ps_ind_05_cat"]) + data["loo_ps_ind_06_bin"])/2.0))/2.0)))
    v["35"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] + data["loo_ps_car_07_cat"]) + (data["loo_ps_ind_07_bin"] + data["loo_ps_car_11_cat"]))/2.0) * 19.500000))
    v["36"] = 0.020000*np.tanh((((data["loo_ps_ind_06_bin"] + (data["loo_ps_car_09_cat"] + data["loo_ps_ind_05_cat"])) + data["loo_ps_ind_16_bin"]) * 19.500000))
    v["37"] = 0.020000*np.tanh((((6.0) * (data["loo_ps_car_05_cat"] + (data["loo_ps_car_11_cat"] + data["loo_ps_ind_05_cat"]))) + data["ps_reg_02"]))
    v["38"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] + (data["loo_ps_car_01_cat"] - data["ps_ind_15"])) + ((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_09_bin"])/2.0)))
    v["39"] = 0.020000*np.tanh(((data["ps_reg_03"] + (data["ps_reg_02"] + (data["loo_ps_ind_06_bin"] - -2.0))) + data["ps_ind_01"]))
    v["40"] = 0.020000*np.tanh((data["ps_car_13"] - (data["ps_ind_15"] - ((5.0) * (data["loo_ps_car_09_cat"] + data["loo_ps_ind_09_bin"])))))
    v["41"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + (data["ps_ind_03"] * data["ps_ind_03"])) + data["ps_ind_03"]) * (8.21348762512207031)))
    v["42"] = 0.020000*np.tanh(((11.08931350708007812) * (((data["loo_ps_ind_05_cat"] + data["ps_car_15"]) + data["ps_ind_01"]) + data["loo_ps_car_09_cat"])))
    v["43"] = 0.020000*np.tanh((19.500000 * (data["loo_ps_car_09_cat"] + ((data["loo_ps_ind_04_cat"] - data["ps_ind_15"]) + data["loo_ps_car_05_cat"]))))
    v["44"] = 0.020000*np.tanh((19.500000 * ((data["loo_ps_ind_17_bin"] + data["ps_reg_03"]) + (data["loo_ps_car_07_cat"] * data["loo_ps_car_07_cat"]))))
    v["45"] = 0.020000*np.tanh((data["ps_car_13"] - (data["ps_ind_15"] - ((data["loo_ps_ind_05_cat"] * (10.0)) - data["ps_car_11"]))))
    v["46"] = 0.020000*np.tanh(((2.0 * ((data["loo_ps_car_07_cat"] + data["ps_ind_01"]) + data["loo_ps_ind_09_bin"])) + data["loo_ps_ind_17_bin"]))
    v["47"] = 0.020000*np.tanh((((data["loo_ps_ind_02_cat"] + (data["loo_ps_ind_16_bin"] + data["loo_ps_ind_02_cat"])) - data["loo_ps_ind_18_bin"]) * (10.0)))
    v["48"] = 0.020000*np.tanh((((data["ps_ind_03"] + (data["ps_ind_03"] * data["ps_ind_03"])) * (14.62119007110595703)) * (14.62119007110595703)))
    v["49"] = 0.020000*np.tanh(((6.846150 * (data["loo_ps_car_01_cat"] + data["loo_ps_car_09_cat"])) + (data["ps_car_13"] + 1.135800)))
    v["50"] = 0.020000*np.tanh(((7.46593427658081055) * ((7.46593427658081055) * ((data["ps_ind_03"] * data["ps_ind_03"]) + data["ps_ind_03"]))))
    v["51"] = 0.020000*np.tanh(((data["loo_ps_ind_07_bin"] + (0.965909 - data["ps_ind_15"])) * (3.642860 - data["loo_ps_ind_06_bin"])))
    v["52"] = 0.020000*np.tanh((data["loo_ps_ind_07_bin"] + ((7.95933532714843750) * (data["loo_ps_car_07_cat"] - (data["ps_ind_03"] * data["loo_ps_ind_02_cat"])))))
    v["53"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_06_bin"] + -2.0))) * 6.846150))
    v["54"] = 0.020000*np.tanh(((data["loo_ps_ind_09_bin"] + (data["loo_ps_car_07_cat"] + ((data["ps_reg_03"] + data["ps_car_13"])/2.0))) * (8.96369743347167969)))
    v["55"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] + (data["ps_ind_03"] - (data["ps_ind_03"] * ((7.0) * data["loo_ps_ind_02_cat"])))))
    v["56"] = 0.020000*np.tanh(((4.0) * (data["ps_reg_02"] + (data["ps_reg_03"] + ((data["loo_ps_ind_02_cat"] + data["ps_car_15"])/2.0)))))
    v["57"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] + (data["loo_ps_car_09_cat"] + (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_06_bin"]))) * 19.500000))
    v["58"] = 0.020000*np.tanh((3.642860 * (data["loo_ps_ind_05_cat"] + ((data["loo_ps_car_05_cat"] * data["ps_ind_03"]) - data["ps_car_11"]))))
    v["59"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] + (data["loo_ps_car_07_cat"] + ((data["ps_ind_15"] * data["loo_ps_ind_18_bin"]) * 6.846150))))
    v["60"] = 0.020000*np.tanh((((6.846150 * data["loo_ps_ind_02_cat"]) + (data["ps_ind_01"] + data["loo_ps_ind_07_bin"])) + data["loo_ps_car_01_cat"]))
    v["61"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_05_cat"] * data["ps_car_13"])) + (data["loo_ps_ind_17_bin"] - data["ps_ind_03"])))
    v["62"] = 0.020000*np.tanh((((data["loo_ps_car_03_cat"] + data["ps_car_13"]) + data["loo_ps_car_04_cat"]) * (data["loo_ps_ind_05_cat"] + data["loo_ps_ind_16_bin"])))
    v["63"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] * data["loo_ps_ind_09_bin"]) + (data["loo_ps_ind_16_bin"] * (data["ps_ind_01"] + data["loo_ps_car_01_cat"]))))
    v["64"] = 0.020000*np.tanh((data["loo_ps_car_03_cat"] * (data["missing"] + (data["loo_ps_ind_17_bin"] + (data["ps_ind_01"] + data["loo_ps_car_11_cat"])))))
    v["65"] = 0.020000*np.tanh((((data["ps_reg_02"] + data["loo_ps_ind_06_bin"]) + data["loo_ps_car_11_cat"]) * (data["loo_ps_ind_05_cat"] - data["ps_car_11"])))
    v["66"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - data["ps_reg_01"]) * (data["ps_car_12"] + (data["loo_ps_car_11_cat"] + data["loo_ps_ind_17_bin"]))))
    v["67"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] * data["loo_ps_ind_06_bin"]) + (data["loo_ps_ind_04_cat"] + ((data["loo_ps_ind_08_bin"] + data["ps_car_13"])/2.0))))
    v["68"] = 0.020000*np.tanh(((data["ps_car_13"] * (data["loo_ps_ind_05_cat"] - data["ps_car_11"])) + (data["loo_ps_ind_16_bin"] * data["loo_ps_ind_05_cat"])))
    v["69"] = 0.020000*np.tanh(((data["ps_ind_01"] * (data["loo_ps_car_05_cat"] - data["ps_ind_15"])) + ((data["loo_ps_car_07_cat"] + data["ps_car_13"])/2.0)))
    v["70"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] * 19.500000) * (data["loo_ps_ind_17_bin"] + data["ps_reg_02"])))
    v["71"] = 0.020000*np.tanh((-1.0 + (data["loo_ps_ind_05_cat"] + (data["ps_ind_03"] * (data["ps_ind_03"] - data["loo_ps_ind_02_cat"])))))
    v["72"] = 0.020000*np.tanh((((data["loo_ps_ind_06_bin"] + data["ps_car_13"]) * (data["ps_ind_03"] + data["loo_ps_ind_05_cat"])) - data["ps_ind_03"]))
    v["73"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_16_bin"] * (data["loo_ps_car_09_cat"] + (data["ps_reg_02"] * data["ps_reg_02"])))))
    v["74"] = 0.020000*np.tanh((((data["loo_ps_car_03_cat"] * 3.0) * 3.0) * (data["ps_ind_15"] + data["loo_ps_ind_17_bin"])))
    v["75"] = 0.020000*np.tanh((((data["ps_ind_01"] + data["loo_ps_ind_17_bin"])/2.0) * ((data["loo_ps_car_07_cat"] - data["ps_car_11"]) - data["ps_ind_15"])))
    v["76"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] - (data["loo_ps_car_04_cat"] * data["ps_car_11"])) + data["ps_reg_02"]) - data["loo_ps_car_04_cat"]))
    v["77"] = 0.019992*np.tanh(((((data["ps_ind_03"] - data["ps_ind_15"]) * data["ps_car_12"]) + data["loo_ps_car_07_cat"]) + data["loo_ps_ind_04_cat"]))
    v["78"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((data["loo_ps_car_09_cat"] + data["ps_ind_01"]) + (data["loo_ps_car_03_cat"] + data["ps_ind_01"]))))
    v["79"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] + data["loo_ps_car_09_cat"]) * ((data["loo_ps_ind_07_bin"] + data["loo_ps_ind_16_bin"]) + data["loo_ps_car_04_cat"])))
    v["80"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] + data["loo_ps_car_07_cat"]) * ((data["loo_ps_car_09_cat"] + data["loo_ps_ind_09_bin"]) + data["ps_reg_02"])))
    v["81"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] + (data["loo_ps_car_03_cat"] * data["loo_ps_ind_09_bin"])) + (data["ps_ind_01"] * data["loo_ps_car_04_cat"])))
    v["82"] = 0.020000*np.tanh(((data["loo_ps_car_01_cat"] * data["loo_ps_car_05_cat"]) - (data["ps_ind_15"] * (data["ps_car_13"] - data["loo_ps_ind_06_bin"]))))
    v["83"] = 0.019996*np.tanh(((data["ps_ind_03"] * ((data["loo_ps_ind_09_bin"] - data["ps_reg_01"]) - data["missing"])) - data["ps_car_11"]))
    v["84"] = 0.020000*np.tanh((data["loo_ps_car_02_cat"] + (((data["ps_car_13"] + data["loo_ps_car_09_cat"])/2.0) + (data["loo_ps_car_02_cat"] - data["ps_ind_03"]))))
    v["85"] = 0.020000*np.tanh((data["ps_ind_01"] + (data["ps_reg_01"] * ((data["missing"] - data["ps_ind_01"]) - data["loo_ps_car_01_cat"]))))
    v["86"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] - ((data["ps_reg_03"] + data["loo_ps_ind_09_bin"])/2.0)) * (data["loo_ps_car_01_cat"] - data["ps_ind_15"])))
    v["87"] = 0.020000*np.tanh(((data["loo_ps_ind_07_bin"] + data["loo_ps_ind_04_cat"]) * (data["ps_ind_03"] + ((data["loo_ps_car_05_cat"] + data["loo_ps_car_09_cat"])/2.0))))
    v["88"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] - data["loo_ps_car_04_cat"]) + ((data["loo_ps_car_07_cat"] + data["loo_ps_ind_07_bin"]) * data["loo_ps_ind_17_bin"])))
    v["89"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_17_bin"]) + ((data["loo_ps_ind_05_cat"] - data["loo_ps_ind_02_cat"]) * data["ps_ind_03"])))
    v["90"] = 0.020000*np.tanh((data["ps_ind_03"] - (((data["ps_ind_03"] * data["loo_ps_ind_02_cat"]) + data["loo_ps_ind_02_cat"]) * 6.846150)))
    v["91"] = 0.019980*np.tanh((np.tanh((-(data["loo_ps_car_11_cat"]))) + (data["loo_ps_car_07_cat"] - (data["ps_ind_15"] * data["loo_ps_car_11_cat"]))))
    v["92"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] + ((data["loo_ps_ind_05_cat"] * (data["ps_reg_02"] + data["loo_ps_ind_17_bin"])) - data["loo_ps_ind_05_cat"])))
    v["93"] = 0.019996*np.tanh(((data["loo_ps_ind_02_cat"] + data["loo_ps_ind_02_cat"]) * ((data["loo_ps_car_08_cat"] - data["ps_ind_03"]) + data["loo_ps_car_08_cat"])))
    v["94"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_07_bin"] + data["ps_reg_02"])) + (data["loo_ps_ind_02_cat"] - data["ps_ind_03"]))/2.0))
    v["95"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + data["ps_reg_01"])/2.0) - (data["loo_ps_ind_06_bin"] * (data["ps_car_15"] + data["ps_reg_01"]))))
    v["96"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] + (((data["loo_ps_car_09_cat"] + data["loo_ps_ind_18_bin"])/2.0) * data["loo_ps_ind_17_bin"])) * data["loo_ps_car_09_cat"]))
    v["97"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((data["loo_ps_ind_07_bin"] + data["ps_ind_15"]) + (data["ps_ind_01"] + data["loo_ps_car_09_cat"]))))
    v["98"] = 0.020000*np.tanh((data["ps_reg_02"] + (((data["loo_ps_ind_02_cat"] - data["ps_ind_03"]) * data["loo_ps_ind_02_cat"]) + -1.0)))
    v["99"] = 0.020000*np.tanh((((data["ps_ind_03"] - data["ps_car_15"]) - data["ps_ind_15"]) * ((data["ps_car_15"] + data["ps_ind_03"])/2.0)))
    v["100"] = 0.020000*np.tanh(((((data["ps_reg_03"] + data["loo_ps_car_09_cat"])/2.0) + data["loo_ps_ind_05_cat"]) * (-2.0 + data["loo_ps_ind_05_cat"])))
    v["101"] = 0.020000*np.tanh((((data["ps_car_15"] + data["loo_ps_car_09_cat"])/2.0) - (data["loo_ps_car_03_cat"] * (data["ps_car_15"] - data["loo_ps_ind_09_bin"]))))
    v["102"] = 0.020000*np.tanh((data["ps_ind_01"] + (data["loo_ps_ind_04_cat"] - ((data["ps_car_11"] + data["ps_ind_01"]) * data["ps_ind_01"]))))
    v["103"] = 0.019988*np.tanh((((data["ps_reg_03"] * (data["loo_ps_ind_05_cat"] - data["ps_ind_01"])) + ((data["ps_car_13"] + data["loo_ps_ind_04_cat"])/2.0))/2.0))
    v["104"] = 0.020000*np.tanh((data["loo_ps_car_08_cat"] + (data["loo_ps_car_03_cat"] * ((data["loo_ps_ind_04_cat"] + data["ps_reg_03"]) + data["ps_reg_03"]))))
    v["105"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_17_bin"]) - data["loo_ps_car_07_cat"]) + data["loo_ps_ind_02_cat"]) - data["loo_ps_ind_05_cat"]))
    v["106"] = 0.019996*np.tanh(((data["ps_ind_01"] - (data["ps_car_15"] * data["loo_ps_ind_17_bin"])) + ((data["ps_ind_01"] + data["loo_ps_car_01_cat"])/2.0)))
    v["107"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] + (data["ps_ind_01"] + data["loo_ps_car_09_cat"])) * (data["loo_ps_ind_17_bin"] - data["ps_reg_01"])))
    v["108"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] * (data["ps_car_11"] + ((data["loo_ps_car_09_cat"] + data["ps_ind_03"]) - data["loo_ps_ind_06_bin"]))))
    v["109"] = 0.020000*np.tanh(((data["missing"] + ((data["loo_ps_ind_05_cat"] + data["ps_reg_02"]) * (data["loo_ps_car_11_cat"] * data["loo_ps_car_05_cat"])))/2.0))
    v["110"] = 0.019996*np.tanh((data["ps_reg_03"] * (((data["ps_car_11"] + data["loo_ps_car_09_cat"])/2.0) - ((data["loo_ps_ind_02_cat"] + data["ps_reg_01"])/2.0))))
    v["111"] = 0.019988*np.tanh(((-1.0 + (data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"])) * (data["loo_ps_ind_02_cat"] + data["loo_ps_ind_02_cat"])))
    v["112"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_05_cat"] - (1.526320 - np.tanh((-(data["loo_ps_ind_05_cat"])))))))
    v["113"] = 0.020000*np.tanh((((((data["loo_ps_ind_16_bin"] + data["loo_ps_car_01_cat"])/2.0) + data["ps_ind_03"])/2.0) * (data["missing"] - data["ps_car_11"])))
    v["114"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - data["ps_reg_01"]) * (data["loo_ps_ind_08_bin"] + (data["ps_ind_01"] + data["loo_ps_car_04_cat"]))))
    v["115"] = 0.020000*np.tanh((-((data["ps_reg_02"] * (data["loo_ps_ind_04_cat"] + (data["ps_reg_02"] * data["loo_ps_car_07_cat"]))))))
    v["116"] = 0.020000*np.tanh(((data["ps_ind_03"] + (data["ps_ind_03"] + 3.0)) * (data["ps_ind_03"] + -2.0)))
    v["117"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] - (data["loo_ps_car_04_cat"] * ((data["ps_car_11"] + data["ps_ind_15"]) + 1.871790))))
    v["118"] = 0.020000*np.tanh((data["ps_reg_03"] * (data["ps_reg_03"] - ((0.513158 - data["loo_ps_ind_07_bin"]) * data["loo_ps_ind_16_bin"]))))
    v["119"] = 0.020000*np.tanh(((((data["ps_reg_03"] + data["loo_ps_car_09_cat"]) * (data["loo_ps_ind_05_cat"] - data["ps_reg_01"])) + data["ps_reg_01"])/2.0))
    v["120"] = 0.020000*np.tanh((((((data["missing"] + data["loo_ps_car_04_cat"])/2.0) + data["loo_ps_car_03_cat"])/2.0) * (data["loo_ps_ind_02_cat"] + data["loo_ps_car_11_cat"])))
    v["121"] = 0.019988*np.tanh(((data["ps_ind_03"] * (data["ps_ind_03"] - data["ps_ind_01"])) + (data["ps_ind_03"] - data["ps_ind_01"])))
    v["122"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] * (data["loo_ps_ind_04_cat"] * (data["loo_ps_ind_02_cat"] - 2.800000))) - data["loo_ps_car_10_cat"]))
    v["123"] = 0.020000*np.tanh((data["loo_ps_car_05_cat"] * (data["ps_ind_01"] + ((data["loo_ps_ind_07_bin"] + ((data["loo_ps_ind_04_cat"] + data["ps_ind_03"])/2.0))/2.0))))
    v["124"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] * (data["ps_reg_02"] * data["ps_reg_01"])) - (0.452381 + data["loo_ps_ind_05_cat"])))
    v["125"] = 0.020000*np.tanh((data["loo_ps_car_01_cat"] * ((data["loo_ps_car_01_cat"] + ((-3.0 * data["ps_car_13"]) - data["loo_ps_ind_06_bin"]))/2.0)))
    v["126"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((data["loo_ps_ind_02_cat"] + (data["ps_reg_03"] + (data["loo_ps_ind_09_bin"] - 0.965909)))/2.0)))
    v["127"] = 0.019988*np.tanh(((((data["loo_ps_ind_02_cat"] - data["loo_ps_car_07_cat"]) + (data["loo_ps_car_04_cat"] * data["ps_car_12"]))/2.0) * 0.485294))
    v["128"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] * data["loo_ps_car_09_cat"]) - data["ps_reg_02"]) * ((data["ps_reg_03"] + data["loo_ps_ind_04_cat"])/2.0)))
    v["129"] = 0.020000*np.tanh((((data["ps_reg_03"] + data["ps_reg_03"]) * np.tanh(data["ps_ind_03"])) - data["ps_ind_03"]))
    v["130"] = 0.020000*np.tanh(((data["ps_reg_03"] * data["ps_reg_03"]) * ((data["loo_ps_ind_17_bin"] + data["ps_car_13"])/2.0)))
    v["131"] = 0.020000*np.tanh((data["ps_reg_01"] * (0.273684 - ((data["loo_ps_car_09_cat"] + (data["ps_reg_03"] - data["ps_reg_01"]))/2.0))))
    v["132"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + data["loo_ps_car_05_cat"])/2.0) * ((data["ps_reg_03"] * data["ps_reg_03"]) - 0.760563)))
    v["133"] = 0.020000*np.tanh((((data["loo_ps_ind_02_cat"] * (-(data["ps_ind_03"]))) * (-(data["ps_ind_03"]))) - data["loo_ps_ind_02_cat"]))
    v["134"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + -3.0)/2.0) * ((((data["ps_car_11"] + 1.135800)/2.0) + data["loo_ps_ind_02_cat"])/2.0)))
    v["135"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] * (((2.0 + (data["loo_ps_ind_04_cat"] - data["loo_ps_car_07_cat"]))/2.0) - data["ps_car_13"])))
    v["136"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((((data["loo_ps_ind_04_cat"] + data["ps_reg_03"])/2.0) + np.tanh(data["ps_ind_03"]))/2.0)))
    v["137"] = 0.020000*np.tanh((((data["ps_reg_01"] * (data["ps_reg_01"] - data["loo_ps_ind_18_bin"])) + (data["ps_car_11"] * data["loo_ps_car_06_cat"]))/2.0))
    v["138"] = 0.019996*np.tanh((data["loo_ps_ind_02_cat"] * (-3.0 + (data["ps_ind_03"] * (data["ps_ind_03"] + data["ps_ind_03"])))))
    v["139"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((data["loo_ps_car_09_cat"] * data["loo_ps_car_09_cat"]) + (data["ps_car_13"] * data["loo_ps_car_07_cat"]))))
    v["140"] = 0.020000*np.tanh(((-((data["ps_ind_03"] * data["loo_ps_ind_02_cat"]))) - np.tanh((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_05_cat"]))))
    v["141"] = 0.020000*np.tanh((data["ps_car_13"] - (data["loo_ps_car_04_cat"] - (data["loo_ps_car_07_cat"] + ((data["loo_ps_ind_18_bin"] + data["loo_ps_car_07_cat"])/2.0)))))
    v["142"] = 0.020000*np.tanh((((data["loo_ps_ind_02_cat"] - data["loo_ps_car_07_cat"]) + (data["loo_ps_ind_17_bin"] * (data["ps_ind_03"] - data["ps_car_15"])))/2.0))
    v["143"] = 0.020000*np.tanh(((((data["loo_ps_ind_02_cat"] + data["ps_car_13"])/2.0) - 1.135800) * (data["loo_ps_ind_02_cat"] + data["ps_car_13"])))
    v["144"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] - data["ps_car_14"]) * (data["loo_ps_car_02_cat"] + (data["loo_ps_car_05_cat"] * data["loo_ps_car_07_cat"]))))
    v["145"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] - data["ps_ind_01"]) * (data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_04_cat"] - data["loo_ps_car_07_cat"]))))
    v["146"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_05_cat"] * ((data["loo_ps_ind_05_cat"] + (-2.0 - 0.452381))/2.0))))
    v["147"] = 0.020000*np.tanh((-((((data["loo_ps_ind_04_cat"] + data["loo_ps_ind_04_cat"]) + data["ps_car_15"]) * data["ps_reg_02"]))))
    v["148"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * ((data["loo_ps_ind_02_cat"] * (data["loo_ps_ind_02_cat"] + data["loo_ps_car_08_cat"])) + data["loo_ps_car_08_cat"])))
    v["149"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] * (data["loo_ps_car_03_cat"] * (data["loo_ps_ind_04_cat"] + data["loo_ps_car_04_cat"]))) - data["loo_ps_car_10_cat"]))
    v["150"] = 0.020000*np.tanh(((data["missing"] * ((data["ps_ind_01"] + ((data["loo_ps_ind_02_cat"] + data["ps_ind_01"])/2.0))/2.0)) - data["loo_ps_car_10_cat"]))
    v["151"] = 0.020000*np.tanh((((data["ps_ind_01"] * (data["ps_ind_01"] * data["loo_ps_car_03_cat"])) + ((data["ps_ind_03"] + data["loo_ps_ind_04_cat"])/2.0))/2.0))
    v["152"] = 0.019965*np.tanh((data["ps_reg_01"] - ((data["ps_reg_01"] + data["loo_ps_ind_02_cat"]) * data["ps_ind_03"])))
    v["153"] = 0.020000*np.tanh(((data["ps_ind_03"] * data["loo_ps_ind_05_cat"]) * (-((data["ps_ind_01"] - data["ps_reg_01"])))))
    v["154"] = 0.019988*np.tanh((((((data["loo_ps_ind_04_cat"] + -1.0)/2.0) + data["ps_ind_01"])/2.0) - (data["ps_car_11"] * data["ps_ind_01"])))
    v["155"] = 0.020000*np.tanh((((data["loo_ps_car_07_cat"] + ((data["loo_ps_car_09_cat"] + data["ps_car_12"])/2.0))/2.0) * (data["loo_ps_ind_06_bin"] * data["loo_ps_ind_16_bin"])))
    v["156"] = 0.020000*np.tanh(((data["loo_ps_car_01_cat"] - data["loo_ps_car_06_cat"]) * (data["missing"] + data["loo_ps_ind_08_bin"])))
    v["157"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + (data["ps_car_12"] * ((data["loo_ps_car_01_cat"] - data["loo_ps_car_07_cat"]) - data["loo_ps_car_07_cat"])))/2.0))
    v["158"] = 0.020000*np.tanh((((data["loo_ps_car_01_cat"] * (data["loo_ps_car_01_cat"] * 0.020833)) + (data["ps_car_15"] * data["missing"]))/2.0))
    v["159"] = 0.020000*np.tanh(((0.093750 - data["ps_reg_01"]) * (data["loo_ps_car_09_cat"] * (data["ps_car_12"] + data["loo_ps_car_09_cat"]))))
    v["160"] = 0.020000*np.tanh((((data["loo_ps_ind_09_bin"] + (data["loo_ps_car_09_cat"] * (data["loo_ps_car_05_cat"] * data["loo_ps_car_04_cat"])))/2.0) * data["loo_ps_car_09_cat"]))
    v["161"] = 0.019977*np.tanh((((-((data["loo_ps_car_01_cat"] * data["ps_reg_03"]))) - data["ps_car_12"]) - data["ps_ind_14"]))
    v["162"] = 0.020000*np.tanh(((((data["loo_ps_car_01_cat"] * data["loo_ps_car_01_cat"]) * (data["loo_ps_car_01_cat"] - data["loo_ps_car_07_cat"])) + 0.760563)/2.0))
    v["163"] = 0.019988*np.tanh((((data["loo_ps_ind_02_cat"] + ((data["ps_car_12"] - 0.513158) - data["loo_ps_ind_06_bin"]))/2.0) - data["loo_ps_car_10_cat"]))
    v["164"] = 0.020000*np.tanh((data["ps_car_15"] + (data["loo_ps_ind_02_cat"] * ((data["loo_ps_car_07_cat"] * data["loo_ps_car_07_cat"]) * data["loo_ps_car_01_cat"]))))
    v["165"] = 0.019988*np.tanh((((data["loo_ps_ind_04_cat"] + (data["loo_ps_car_07_cat"] - data["ps_car_15"]))/2.0) + (data["loo_ps_ind_09_bin"] * data["loo_ps_car_07_cat"])))
    v["166"] = 0.020000*np.tanh((((data["loo_ps_car_10_cat"] + data["loo_ps_car_01_cat"]) * data["loo_ps_ind_17_bin"]) * (data["loo_ps_car_09_cat"] + data["loo_ps_ind_04_cat"])))
    v["167"] = 0.018984*np.tanh((((10.63927078247070312) * (-(data["loo_ps_ind_05_cat"]))) * ((data["ps_reg_01"] + np.tanh(data["loo_ps_ind_16_bin"]))/2.0)))
    v["168"] = 0.020000*np.tanh((((data["loo_ps_car_07_cat"] + data["ps_ind_03"])/2.0) * (((data["loo_ps_ind_17_bin"] * data["loo_ps_ind_04_cat"]) + data["ps_ind_03"])/2.0)))
    v["169"] = 0.020000*np.tanh(((((-1.0 + (data["loo_ps_ind_05_cat"] - 0.347826))/2.0) + (data["ps_ind_15"] * data["ps_reg_01"]))/2.0))
    v["170"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * ((0.100000 * data["loo_ps_car_01_cat"]) - ((data["loo_ps_car_10_cat"] + 0.166667)/2.0))))
    v["171"] = 0.020000*np.tanh(((data["loo_ps_car_05_cat"] * data["loo_ps_ind_17_bin"]) * ((data["loo_ps_ind_04_cat"] * data["loo_ps_ind_04_cat"]) - data["ps_car_11"])))
    v["172"] = 0.019992*np.tanh((data["loo_ps_ind_07_bin"] * (data["ps_reg_03"] + (-(((data["loo_ps_car_01_cat"] + np.tanh(data["loo_ps_car_02_cat"]))/2.0))))))
    v["173"] = 0.020000*np.tanh((((((data["loo_ps_ind_04_cat"] + data["loo_ps_ind_05_cat"])/2.0) + data["loo_ps_ind_08_bin"])/2.0) * (data["loo_ps_car_09_cat"] + data["ps_ind_03"])))
    v["174"] = 0.019988*np.tanh(((((data["loo_ps_ind_04_cat"] + data["ps_ind_14"])/2.0) - ((data["loo_ps_car_03_cat"] + 1.526320)/2.0)) * data["loo_ps_ind_04_cat"]))
    v["175"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] * ((((data["loo_ps_ind_02_cat"] + (-(data["loo_ps_ind_12_bin"])))/2.0) + data["loo_ps_car_05_cat"])/2.0)))
    v["176"] = 0.019902*np.tanh(((data["ps_reg_03"] + (data["ps_reg_03"] + np.tanh(data["loo_ps_ind_05_cat"]))) * (-(data["loo_ps_car_11_cat"]))))
    v["177"] = 0.019996*np.tanh((((data["ps_reg_02"] + data["loo_ps_ind_02_cat"])/2.0) * (data["ps_car_12"] + (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_16_bin"]))))
    v["178"] = 0.020000*np.tanh((data["ps_ind_15"] * (data["ps_ind_03"] * (-(((data["ps_reg_01"] + data["ps_ind_01"])/2.0))))))
    v["179"] = 0.020000*np.tanh(((data["loo_ps_car_01_cat"] * data["loo_ps_car_11_cat"]) * (data["ps_ind_03"] + ((data["ps_reg_01"] + data["loo_ps_car_01_cat"])/2.0))))
    v["180"] = 0.020000*np.tanh((((data["loo_ps_ind_04_cat"] + (-(data["loo_ps_ind_16_bin"])))/2.0) - (data["ps_reg_03"] * data["ps_reg_01"])))
    v["181"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] + (data["loo_ps_car_04_cat"] * (0.583333 - (data["ps_ind_03"] * data["ps_ind_03"]))))/2.0))
    v["182"] = 0.019992*np.tanh(((data["ps_reg_03"] * data["ps_reg_03"]) + np.tanh((data["ps_ind_03"] * data["loo_ps_ind_04_cat"]))))
    v["183"] = 0.019961*np.tanh(((data["ps_ind_15"] * (np.tanh((-(data["loo_ps_car_11_cat"]))) - data["loo_ps_car_09_cat"])) * data["loo_ps_car_11_cat"]))
    v["184"] = 0.020000*np.tanh((((data["loo_ps_ind_16_bin"] + data["loo_ps_ind_07_bin"])/2.0) * ((data["missing"] + (-(data["ps_reg_01"])))/2.0)))
    v["185"] = 0.019949*np.tanh((data["loo_ps_car_04_cat"] * ((data["ps_reg_03"] + (data["loo_ps_ind_02_cat"] - data["loo_ps_car_04_cat"]))/2.0)))
    v["186"] = 0.020000*np.tanh((((data["ps_car_12"] + data["loo_ps_car_03_cat"])/2.0) * (((data["ps_car_12"] + data["loo_ps_ind_05_cat"])/2.0) * data["loo_ps_car_04_cat"])))
    v["187"] = 0.020000*np.tanh((np.tanh(data["loo_ps_ind_17_bin"]) * (data["loo_ps_ind_05_cat"] * ((data["loo_ps_ind_04_cat"] + (-(data["ps_ind_15"])))/2.0))))
    v["188"] = 0.020000*np.tanh((((data["ps_ind_03"] + data["loo_ps_ind_12_bin"])/2.0) * (data["ps_ind_03"] - np.tanh(data["ps_ind_03"]))))
    v["189"] = 0.020000*np.tanh(((((data["loo_ps_car_11_cat"] + data["loo_ps_ind_04_cat"])/2.0) * data["loo_ps_ind_17_bin"]) * (data["loo_ps_ind_02_cat"] - data["loo_ps_car_07_cat"])))
    v["190"] = 0.020000*np.tanh(((data["ps_ind_01"] * data["ps_ind_01"]) * (((data["loo_ps_car_01_cat"] + data["ps_ind_03"])/2.0) * data["loo_ps_ind_17_bin"])))
    v["191"] = 0.020000*np.tanh((((data["loo_ps_ind_02_cat"] + (-(data["ps_car_15"])))/2.0) * data["ps_car_15"]))
    v["192"] = 0.020000*np.tanh(((0.166667 + (((data["ps_car_12"] + data["loo_ps_ind_17_bin"])/2.0) * (data["loo_ps_ind_04_cat"] * data["loo_ps_ind_04_cat"])))/2.0))
    v["193"] = 0.019980*np.tanh((data["loo_ps_car_02_cat"] * (((data["loo_ps_ind_13_bin"] + data["loo_ps_car_02_cat"])/2.0) - (data["ps_reg_02"] + data["loo_ps_car_06_cat"]))))
    v["194"] = 0.019895*np.tanh(((data["ps_reg_03"] * (-((data["ps_ind_03"] * data["ps_ind_15"])))) * data["loo_ps_car_01_cat"]))
    v["195"] = 0.020000*np.tanh((-(((((data["loo_ps_ind_04_cat"] + data["ps_ind_01"])/2.0) * data["ps_ind_01"]) * data["ps_ind_03"]))))
    v["196"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] * (-(data["ps_car_11"]))) * (data["loo_ps_car_02_cat"] + data["loo_ps_car_01_cat"])))
    v["197"] = 0.020000*np.tanh((np.tanh(data["loo_ps_car_01_cat"]) * (data["ps_ind_14"] + (-((data["loo_ps_ind_18_bin"] * data["loo_ps_car_09_cat"]))))))
    v["198"] = 0.018046*np.tanh((19.500000 * (0.583333 - (6.846150 * data["loo_ps_car_10_cat"]))))
    v["199"] = 0.019988*np.tanh((-2.0 + (((data["loo_ps_car_10_cat"] + 0.965909) + data["ps_car_11"]) * data["ps_car_11"])))
    v["200"] = 0.020000*np.tanh(((data["ps_ind_03"] + ((data["ps_ind_03"] * data["ps_ind_01"]) * (data["ps_car_13"] - data["ps_ind_01"])))/2.0))
    v["201"] = 0.019984*np.tanh((data["ps_reg_02"] * ((((-(data["loo_ps_car_01_cat"])) + data["loo_ps_ind_05_cat"])/2.0) * data["loo_ps_ind_17_bin"])))
    v["202"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((((0.965909 - data["loo_ps_car_01_cat"]) + data["loo_ps_ind_02_cat"])/2.0) * data["ps_ind_01"])))
    v["203"] = 0.020000*np.tanh((data["ps_reg_03"] * ((data["ps_ind_15"] * (data["ps_ind_15"] + data["loo_ps_ind_09_bin"])) + data["ps_ind_15"])))
    v["204"] = 0.020000*np.tanh((((data["loo_ps_car_10_cat"] + data["loo_ps_car_03_cat"])/2.0) * (-(((data["loo_ps_ind_18_bin"] + data["ps_car_15"])/2.0)))))
    v["205"] = 0.020000*np.tanh((((-(np.tanh(data["loo_ps_ind_02_cat"]))) + (((data["ps_ind_03"] + data["loo_ps_ind_02_cat"])/2.0) * data["loo_ps_ind_16_bin"]))/2.0))
    v["206"] = 0.020000*np.tanh(np.tanh((19.500000 * np.tanh((data["ps_ind_14"] + (0.965909 - data["ps_car_15"]))))))
    v["207"] = 0.019945*np.tanh(((19.500000 - (data["loo_ps_ind_02_cat"] * 19.500000)) * (data["ps_reg_02"] - 2.800000)))
    v["208"] = 0.019969*np.tanh((((data["ps_reg_02"] * (data["loo_ps_car_01_cat"] * data["loo_ps_ind_12_bin"])) + (data["loo_ps_car_11_cat"] * data["loo_ps_car_08_cat"]))/2.0))
    v["209"] = 0.019996*np.tanh((((data["ps_ind_03"] + data["loo_ps_ind_12_bin"])/2.0) * ((data["ps_ind_03"] + (data["missing"] - data["loo_ps_ind_02_cat"]))/2.0)))
    v["210"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] * (data["loo_ps_ind_04_cat"] * (data["ps_reg_03"] * (-(data["loo_ps_car_07_cat"]))))))
    v["211"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * (data["ps_reg_03"] + ((data["loo_ps_ind_02_cat"] * data["loo_ps_ind_09_bin"]) - 0.600000))))
    v["212"] = 0.017421*np.tanh(np.tanh(((-((data["ps_car_11"] + data["loo_ps_ind_08_bin"]))) * data["loo_ps_ind_05_cat"])))
    v["213"] = 0.019305*np.tanh((((data["loo_ps_ind_02_cat"] * data["loo_ps_car_04_cat"]) - ((data["missing"] + data["loo_ps_car_04_cat"])/2.0)) - data["loo_ps_ind_02_cat"]))
    v["214"] = 0.019992*np.tanh((data["ps_car_12"] * (((-(data["ps_car_11"])) + (data["ps_reg_01"] + data["loo_ps_ind_04_cat"]))/2.0)))
    v["215"] = 0.019879*np.tanh((data["loo_ps_ind_05_cat"] * (data["ps_car_13"] * ((data["ps_car_14"] * data["ps_ind_15"]) + data["loo_ps_ind_05_cat"]))))
    v["216"] = 0.019984*np.tanh(((data["loo_ps_car_01_cat"] * data["loo_ps_ind_17_bin"]) * (data["loo_ps_ind_02_cat"] - data["loo_ps_ind_17_bin"])))
    v["217"] = 0.020000*np.tanh((((data["ps_reg_03"] * data["loo_ps_car_08_cat"]) - np.tanh(np.tanh(data["loo_ps_car_11_cat"]))) * data["loo_ps_ind_05_cat"]))
    v["218"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] + (0.273684 - data["loo_ps_ind_12_bin"])) * (data["ps_reg_02"] * data["loo_ps_car_07_cat"])))
    v["219"] = 0.020000*np.tanh(((((data["loo_ps_ind_16_bin"] + (data["loo_ps_ind_06_bin"] * 0.166667))/2.0) * data["loo_ps_ind_06_bin"]) * data["loo_ps_ind_05_cat"]))
    v["220"] = 0.020000*np.tanh(((data["ps_ind_15"] * data["ps_car_15"]) * (-(data["loo_ps_ind_17_bin"]))))
    v["221"] = 0.020000*np.tanh(((((data["loo_ps_car_02_cat"] * data["ps_ind_03"]) + (data["loo_ps_car_02_cat"] * data["ps_reg_01"]))/2.0) * data["ps_ind_01"]))
    v["222"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (data["loo_ps_ind_18_bin"] * (data["loo_ps_ind_07_bin"] - (data["loo_ps_ind_04_cat"] * data["loo_ps_ind_16_bin"])))))
    v["223"] = 0.020000*np.tanh(((data["loo_ps_ind_12_bin"] * data["ps_car_11"]) + (data["loo_ps_ind_04_cat"] * (data["ps_reg_01"] * data["loo_ps_car_08_cat"]))))
    v["224"] = 0.019988*np.tanh((data["ps_car_14"] * ((data["ps_car_14"] + ((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_05_cat"]) * data["loo_ps_car_01_cat"]))/2.0)))
    v["225"] = 0.019090*np.tanh((data["ps_ind_15"] * (data["loo_ps_ind_07_bin"] - (-((data["loo_ps_car_06_cat"] * data["ps_ind_15"]))))))
    v["226"] = 0.015734*np.tanh((data["ps_ind_15"] - (6.846150 * ((data["loo_ps_car_11_cat"] + data["loo_ps_car_06_cat"]) + data["loo_ps_car_06_cat"]))))
    v["227"] = 0.020000*np.tanh(((data["ps_car_12"] - data["loo_ps_car_06_cat"]) * (data["loo_ps_ind_02_cat"] * (-(data["loo_ps_car_08_cat"])))))
    v["228"] = 0.019996*np.tanh(((data["loo_ps_ind_09_bin"] * data["ps_car_12"]) - np.tanh(((data["ps_reg_02"] + data["ps_car_12"])/2.0))))
    v["229"] = 0.019219*np.tanh((data["ps_ind_01"] * ((data["loo_ps_ind_17_bin"] * data["loo_ps_car_06_cat"]) - np.tanh(data["loo_ps_ind_17_bin"]))))
    v["230"] = 0.019996*np.tanh((data["loo_ps_ind_07_bin"] * (data["ps_car_13"] * (data["loo_ps_car_11_cat"] - np.tanh(2.0)))))
    v["231"] = 0.020000*np.tanh((data["loo_ps_car_11_cat"] * ((data["ps_ind_14"] * data["ps_ind_15"]) * data["ps_car_12"])))
    v["232"] = 0.019816*np.tanh(((np.tanh(data["loo_ps_ind_17_bin"]) * data["missing"]) * (np.tanh(data["ps_car_15"]) - data["loo_ps_car_02_cat"])))
    v["233"] = 0.020000*np.tanh((((data["loo_ps_car_01_cat"] - data["ps_reg_01"]) * data["loo_ps_car_10_cat"]) * (data["ps_ind_03"] + data["loo_ps_ind_17_bin"])))
    v["234"] = 0.017378*np.tanh(((data["ps_ind_01"] * data["loo_ps_car_04_cat"]) * (data["loo_ps_car_01_cat"] + (data["ps_car_12"] * data["loo_ps_car_04_cat"]))))
    v["235"] = 0.020000*np.tanh((data["ps_reg_03"] * ((((data["ps_reg_03"] + data["loo_ps_ind_05_cat"])/2.0) - data["loo_ps_ind_02_cat"]) - data["ps_reg_01"])))
    v["236"] = 0.020000*np.tanh((data["ps_ind_01"] * (((data["loo_ps_car_09_cat"] + data["loo_ps_ind_04_cat"])/2.0) * ((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"])/2.0))))
    v["237"] = 0.019996*np.tanh((data["ps_ind_01"] * ((data["loo_ps_ind_02_cat"] + (data["ps_reg_01"] + (-(data["loo_ps_car_11_cat"]))))/2.0)))
    v["238"] = 0.019941*np.tanh((-((data["loo_ps_car_08_cat"] * (data["loo_ps_car_02_cat"] - ((data["loo_ps_car_04_cat"] + (-(data["loo_ps_ind_18_bin"])))/2.0))))))
    v["239"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (((data["loo_ps_ind_17_bin"] * (-(np.tanh(data["loo_ps_car_08_cat"])))) + data["ps_reg_02"])/2.0)))
    v["240"] = 0.020000*np.tanh(((np.tanh((-(data["loo_ps_ind_08_bin"]))) + (data["ps_reg_03"] * (data["loo_ps_ind_08_bin"] * data["ps_reg_01"])))/2.0))
    v["241"] = 0.019969*np.tanh((((data["loo_ps_car_02_cat"] + data["ps_car_15"])/2.0) * ((-(data["loo_ps_ind_17_bin"])) - data["loo_ps_car_10_cat"])))
    v["242"] = 0.019371*np.tanh(((data["loo_ps_car_03_cat"] * (data["ps_ind_14"] - data["loo_ps_ind_05_cat"])) * (data["ps_car_11"] + data["ps_car_11"])))
    v["243"] = 0.020000*np.tanh((((data["loo_ps_car_10_cat"] * (data["loo_ps_ind_17_bin"] * data["loo_ps_ind_04_cat"])) + (-(np.tanh(data["loo_ps_ind_04_cat"]))))/2.0))
    v["244"] = 0.020000*np.tanh((np.tanh((19.500000 * ((0.452381 + data["loo_ps_ind_18_bin"])/2.0))) * data["loo_ps_ind_05_cat"]))
    v["245"] = 0.019996*np.tanh((((data["loo_ps_car_01_cat"] * data["loo_ps_ind_02_cat"]) * (data["loo_ps_car_08_cat"] * data["ps_reg_01"])) * (4.34056949615478516)))
    v["246"] = 0.019988*np.tanh(((data["loo_ps_car_01_cat"] + data["loo_ps_ind_16_bin"]) * (data["loo_ps_ind_02_cat"] - data["loo_ps_ind_05_cat"])))
    v["247"] = 0.020000*np.tanh((data["ps_car_14"] * (((data["loo_ps_car_07_cat"] * data["ps_ind_15"]) + ((data["ps_car_13"] + data["loo_ps_ind_13_bin"])/2.0))/2.0)))
    v["248"] = 0.019969*np.tanh((data["ps_reg_02"] * ((data["ps_ind_15"] + data["ps_ind_14"]) * (data["loo_ps_ind_08_bin"] * data["loo_ps_car_08_cat"]))))
    v["249"] = 0.019984*np.tanh((data["loo_ps_ind_02_cat"] * (data["missing"] - ((data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"]) * data["loo_ps_car_08_cat"]))))
    v["250"] = 0.020000*np.tanh((((((data["loo_ps_car_03_cat"] + data["ps_car_15"])/2.0) + data["loo_ps_ind_06_bin"])/2.0) * (data["ps_reg_02"] * data["ps_car_15"])))
    v["251"] = 0.020000*np.tanh(((data["ps_car_11"] * ((data["loo_ps_car_06_cat"] + (data["ps_reg_02"] * data["loo_ps_ind_12_bin"]))/2.0)) - data["loo_ps_car_10_cat"]))
    v["252"] = 0.019453*np.tanh((data["ps_ind_03"] * ((0.273684 * data["loo_ps_ind_05_cat"]) + (0.020833 - data["ps_car_14"]))))
    v["253"] = 0.020000*np.tanh((data["ps_car_13"] * ((data["ps_car_13"] * (data["loo_ps_ind_12_bin"] * data["ps_ind_15"])) + data["ps_ind_14"])))
    v["254"] = 0.020000*np.tanh((-((((data["loo_ps_car_10_cat"] * (data["ps_car_13"] + data["ps_car_13"])) + np.tanh(data["loo_ps_ind_05_cat"]))/2.0))))
    v["255"] = 0.019961*np.tanh(((((data["ps_ind_03"] + data["loo_ps_ind_07_bin"])/2.0) * data["ps_ind_14"]) - (data["ps_reg_02"] * data["loo_ps_car_10_cat"])))
    v["256"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * ((data["loo_ps_car_11_cat"] * (data["loo_ps_ind_04_cat"] - 0.583333)) - data["loo_ps_ind_08_bin"])))
    v["257"] = 0.019664*np.tanh((data["loo_ps_car_07_cat"] * (data["loo_ps_ind_09_bin"] - data["loo_ps_car_01_cat"])))
    v["258"] = 0.020000*np.tanh(((-(data["loo_ps_ind_17_bin"])) * np.tanh((data["loo_ps_car_11_cat"] * (data["loo_ps_car_08_cat"] + data["ps_ind_15"])))))
    v["259"] = 0.019644*np.tanh(((data["ps_ind_01"] * (-(data["ps_ind_03"]))) * (data["loo_ps_ind_04_cat"] + data["loo_ps_ind_04_cat"])))
    v["260"] = 0.020000*np.tanh((data["ps_reg_03"] * (-((data["ps_car_15"] * (data["loo_ps_car_07_cat"] * data["loo_ps_ind_17_bin"]))))))
    v["261"] = 0.020000*np.tanh((data["ps_car_12"] * ((data["ps_car_12"] * (data["ps_ind_15"] * data["loo_ps_ind_04_cat"])) + data["ps_ind_15"])))
    v["262"] = 0.018953*np.tanh(((data["ps_ind_15"] + (data["ps_ind_15"] * ((data["loo_ps_ind_08_bin"] - data["loo_ps_car_04_cat"]) - data["loo_ps_car_03_cat"])))/2.0))
    v["263"] = 0.020000*np.tanh((0.166667 * (((data["ps_car_13"] + data["loo_ps_ind_04_cat"])/2.0) - (data["ps_car_13"] * data["loo_ps_car_03_cat"]))))
    v["264"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * ((data["loo_ps_ind_09_bin"] + ((data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"]) + data["loo_ps_ind_10_bin"]))/2.0)))
    v["265"] = 0.019992*np.tanh((data["ps_ind_14"] * ((data["loo_ps_car_03_cat"] * (data["loo_ps_ind_07_bin"] + data["ps_ind_15"])) * data["loo_ps_car_11_cat"])))
    v["266"] = 0.019988*np.tanh((np.tanh(np.tanh(data["loo_ps_car_07_cat"])) + (np.tanh(data["ps_ind_01"]) * (-(data["ps_ind_14"])))))
    v["267"] = 0.020000*np.tanh((np.tanh(np.tanh(data["ps_ind_01"])) * ((data["ps_car_14"] + (data["loo_ps_ind_04_cat"] - data["loo_ps_car_01_cat"]))/2.0)))
    v["268"] = 0.020000*np.tanh((((data["loo_ps_ind_02_cat"] * (data["ps_ind_03"] * data["ps_ind_03"])) - data["loo_ps_ind_02_cat"]) - data["loo_ps_ind_02_cat"]))
    v["269"] = 0.018969*np.tanh((data["loo_ps_ind_05_cat"] * (data["loo_ps_car_07_cat"] * (data["ps_reg_02"] + data["ps_car_12"]))))
    v["270"] = 0.019996*np.tanh((data["ps_car_13"] * (data["loo_ps_ind_12_bin"] * (1.135800 + (data["ps_ind_15"] + data["loo_ps_ind_11_bin"])))))
    v["271"] = 0.019996*np.tanh((-((data["loo_ps_ind_05_cat"] * ((data["loo_ps_car_07_cat"] * (-(data["loo_ps_car_09_cat"]))) + data["loo_ps_car_07_cat"])))))
    v["272"] = 0.020000*np.tanh((data["ps_car_11"] * ((-(data["ps_reg_03"])) * (data["loo_ps_ind_05_cat"] * np.tanh(-1.0)))))
    v["273"] = 0.020000*np.tanh((((data["loo_ps_ind_09_bin"] + data["loo_ps_ind_17_bin"])/2.0) * ((data["loo_ps_ind_18_bin"] + (data["ps_reg_02"] - data["ps_ind_14"]))/2.0)))
    v["274"] = 0.020000*np.tanh((data["ps_ind_01"] * (((data["ps_ind_01"] * data["loo_ps_car_03_cat"]) + data["loo_ps_ind_02_cat"])/2.0)))
    v["275"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (data["ps_ind_01"] * (data["loo_ps_ind_06_bin"] * (-(data["ps_ind_03"]))))))
    v["276"] = 0.019949*np.tanh(((np.tanh((((data["loo_ps_ind_12_bin"] + data["ps_reg_03"])/2.0) * 19.500000)) + (-(data["ps_reg_03"])))/2.0))
    v["277"] = 0.016070*np.tanh(((data["ps_car_15"] + data["loo_ps_car_03_cat"]) * (-((data["ps_reg_03"] + data["loo_ps_ind_05_cat"])))))
    v["278"] = 0.019930*np.tanh(((data["loo_ps_ind_02_cat"] + ((data["loo_ps_car_05_cat"] * data["loo_ps_ind_05_cat"]) - data["loo_ps_ind_05_cat"]))/2.0))
    v["279"] = 0.017035*np.tanh((((data["loo_ps_car_09_cat"] * data["ps_reg_03"]) + data["loo_ps_ind_02_cat"]) * data["loo_ps_ind_05_cat"]))
    v["280"] = 0.019965*np.tanh(((data["ps_reg_01"] + data["loo_ps_ind_07_bin"]) * ((data["loo_ps_ind_05_cat"] + ((data["loo_ps_car_11_cat"] + data["loo_ps_car_08_cat"])/2.0))/2.0)))
    v["281"] = 0.020000*np.tanh((-(((np.tanh(((data["loo_ps_car_01_cat"] + data["loo_ps_car_05_cat"])/2.0)) + (data["loo_ps_car_10_cat"] * data["ps_car_15"]))/2.0))))
    v["282"] = 0.019992*np.tanh(((data["ps_ind_03"] - data["loo_ps_ind_18_bin"]) * (((data["ps_ind_03"] + data["ps_car_12"])/2.0) * data["loo_ps_car_02_cat"])))
    v["283"] = 0.019945*np.tanh((data["loo_ps_ind_04_cat"] * (-((data["ps_reg_02"] * (data["loo_ps_car_07_cat"] * data["loo_ps_ind_17_bin"]))))))
    v["284"] = 0.020000*np.tanh(((-((3.642860 + (-(data["loo_ps_car_09_cat"]))))) * (data["ps_ind_01"] * data["loo_ps_car_09_cat"])))
    v["285"] = 0.018918*np.tanh((data["ps_ind_01"] * (data["ps_ind_01"] - (data["ps_ind_01"] * data["ps_ind_01"]))))
    v["286"] = 0.018738*np.tanh((((data["ps_ind_03"] + data["ps_ind_15"])/2.0) - (data["ps_ind_15"] * (data["ps_ind_15"] * data["ps_ind_03"]))))
    v["287"] = 0.019957*np.tanh(((data["loo_ps_car_08_cat"] * data["loo_ps_ind_08_bin"]) * ((data["loo_ps_car_07_cat"] + data["missing"])/2.0)))
    v["288"] = 0.019598*np.tanh(((data["loo_ps_ind_09_bin"] * ((data["ps_ind_01"] + (-(data["loo_ps_car_05_cat"])))/2.0)) + data["loo_ps_ind_13_bin"]))
    v["289"] = 0.020000*np.tanh((((data["loo_ps_car_01_cat"] * data["ps_ind_15"]) + data["ps_car_11"]) * data["loo_ps_ind_12_bin"]))
    v["290"] = 0.020000*np.tanh((((data["loo_ps_ind_02_cat"] - (data["ps_car_11"] * data["loo_ps_car_09_cat"])) * data["loo_ps_ind_17_bin"]) * data["loo_ps_car_10_cat"]))
    v["291"] = 0.019301*np.tanh((data["loo_ps_car_04_cat"] * ((np.tanh(data["loo_ps_car_04_cat"]) - data["ps_ind_03"]) * data["ps_ind_03"])))
    v["292"] = 0.020000*np.tanh(((data["ps_reg_03"] - data["ps_car_11"]) * ((np.tanh(data["loo_ps_car_04_cat"]) + (-(data["ps_car_14"])))/2.0)))
    v["293"] = 0.019086*np.tanh((((data["ps_ind_03"] * data["loo_ps_ind_12_bin"]) + ((data["loo_ps_ind_12_bin"] - data["loo_ps_car_06_cat"]) * data["ps_reg_03"]))/2.0))
    v["294"] = 0.020000*np.tanh((data["loo_ps_ind_12_bin"] - (data["loo_ps_ind_18_bin"] * (data["loo_ps_ind_04_cat"] * (-(data["ps_reg_03"]))))))
    v["295"] = 0.019988*np.tanh(((((data["ps_reg_03"] * data["ps_car_12"]) + data["loo_ps_ind_02_cat"])/2.0) * np.tanh(data["loo_ps_car_04_cat"])))
    v["296"] = 0.019898*np.tanh((data["loo_ps_car_07_cat"] * (-((((data["ps_car_11"] * (-(data["loo_ps_car_08_cat"]))) + data["loo_ps_car_09_cat"])/2.0)))))
    v["297"] = 0.019973*np.tanh(((data["loo_ps_ind_02_cat"] * (data["loo_ps_ind_16_bin"] + (-(data["loo_ps_car_05_cat"])))) * (-(data["loo_ps_car_08_cat"]))))
    v["298"] = 0.019980*np.tanh((((data["loo_ps_ind_18_bin"] * (data["loo_ps_ind_04_cat"] * data["loo_ps_ind_16_bin"])) - data["ps_car_14"]) * data["ps_ind_14"]))
    v["299"] = 0.016964*np.tanh((((-(0.347826)) + np.tanh((data["loo_ps_car_06_cat"] - data["ps_reg_01"])))/2.0))
    v["300"] = 0.020000*np.tanh(((data["ps_ind_15"] * data["ps_car_15"]) * ((data["loo_ps_ind_12_bin"] + (-(data["ps_car_15"])))/2.0)))
    v["301"] = 0.019957*np.tanh(((data["ps_car_14"] * data["loo_ps_ind_04_cat"]) * (data["ps_ind_01"] + (data["ps_car_14"] * data["loo_ps_car_08_cat"]))))
    v["302"] = 0.020000*np.tanh((data["ps_ind_14"] * (data["ps_car_11"] * (data["ps_car_12"] + (data["ps_ind_03"] * data["ps_car_12"])))))
    v["303"] = 0.020000*np.tanh((data["ps_reg_01"] * (data["loo_ps_car_01_cat"] * ((data["loo_ps_car_01_cat"] + (data["ps_ind_03"] * data["ps_reg_02"]))/2.0))))
    v["304"] = 0.019594*np.tanh(((data["loo_ps_ind_04_cat"] * data["loo_ps_ind_02_cat"]) * (data["loo_ps_car_06_cat"] * (0.347826 + 2.800000))))
    v["305"] = 0.019992*np.tanh((data["ps_reg_01"] * ((data["loo_ps_car_02_cat"] - data["loo_ps_car_04_cat"]) * data["loo_ps_ind_05_cat"])))
    v["306"] = 0.020000*np.tanh(np.tanh(((((data["loo_ps_ind_09_bin"] * data["loo_ps_ind_05_cat"]) + data["ps_ind_03"])/2.0) * (-(data["ps_reg_01"])))))
    v["307"] = 0.018308*np.tanh((-((19.500000 * np.tanh((-1.0 - (data["loo_ps_ind_11_bin"] * 19.500000)))))))
    v["308"] = 0.019754*np.tanh(((1.0 + data["ps_reg_02"]) * ((data["ps_reg_02"] - 3.0) * 3.0)))
    v["309"] = 0.019265*np.tanh((data["loo_ps_ind_04_cat"] * ((data["ps_ind_01"] * (data["ps_reg_01"] * data["ps_ind_01"])) * data["ps_ind_01"])))
    v["310"] = 0.019879*np.tanh((((data["ps_reg_03"] + data["ps_ind_03"])/2.0) * np.tanh((data["ps_reg_03"] * data["ps_ind_03"]))))
    v["311"] = 0.020000*np.tanh(((data["ps_reg_02"] * (data["loo_ps_ind_11_bin"] - (data["ps_car_14"] * data["loo_ps_car_04_cat"]))) * data["ps_reg_01"]))
    v["312"] = 0.019965*np.tanh((data["ps_reg_03"] * ((data["ps_ind_15"] + ((data["ps_ind_15"] * data["loo_ps_ind_09_bin"]) - data["loo_ps_ind_08_bin"]))/2.0)))
    v["313"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] * ((data["loo_ps_ind_02_cat"] + (data["ps_car_11"] * data["loo_ps_car_06_cat"]))/2.0)))
    v["314"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_11_bin"] - (data["loo_ps_ind_04_cat"] * data["ps_car_11"]))))
    v["315"] = 0.019461*np.tanh((data["loo_ps_car_03_cat"] * (data["loo_ps_ind_05_cat"] * (data["loo_ps_car_02_cat"] + data["loo_ps_ind_04_cat"]))))
    v["316"] = 0.014366*np.tanh(np.tanh(np.tanh((data["ps_reg_01"] - ((data["ps_ind_03"] + ((data["ps_ind_14"] + data["loo_ps_car_10_cat"])/2.0))/2.0)))))
    v["317"] = 0.019996*np.tanh((data["ps_car_12"] * ((data["loo_ps_ind_12_bin"] + (data["ps_car_11"] * (data["ps_ind_15"] + data["ps_car_11"])))/2.0)))
    v["318"] = 0.020000*np.tanh(np.tanh((data["ps_car_14"] * ((-(np.tanh(data["ps_car_14"]))) + (-(data["ps_reg_03"]))))))
    v["319"] = 0.015050*np.tanh((-((np.tanh(data["loo_ps_car_04_cat"]) * (data["ps_car_15"] * (data["loo_ps_ind_18_bin"] - data["loo_ps_ind_06_bin"]))))))
    v["320"] = 0.020000*np.tanh((data["ps_car_13"] * (((data["loo_ps_car_04_cat"] * (data["ps_car_14"] - data["ps_ind_03"])) + data["ps_ind_03"])/2.0)))
    v["321"] = 0.020000*np.tanh((-((data["ps_car_14"] + ((0.347826 - data["ps_car_14"]) * data["ps_car_15"])))))
    v["322"] = 0.019977*np.tanh(((0.166667 + (data["ps_ind_14"] * ((data["ps_reg_03"] - data["loo_ps_ind_08_bin"]) - data["loo_ps_ind_17_bin"])))/2.0))
    v["323"] = 0.020000*np.tanh((data["missing"] * (-((data["loo_ps_car_09_cat"] * ((data["ps_car_12"] + data["loo_ps_ind_18_bin"])/2.0))))))
    v["324"] = 0.018195*np.tanh((((data["loo_ps_ind_09_bin"] + (-(np.tanh(data["loo_ps_ind_05_cat"]))))/2.0) * (data["loo_ps_ind_18_bin"] + data["loo_ps_car_07_cat"])))
    v["325"] = 0.019359*np.tanh((-((19.500000 * ((0.485294 + np.tanh((data["loo_ps_ind_11_bin"] * 19.500000)))/2.0)))))
    v["326"] = 0.019992*np.tanh((-3.0 - ((-((data["ps_ind_03"] * data["ps_ind_03"]))) + data["ps_ind_03"])))
    v["327"] = 0.020000*np.tanh((np.tanh((data["ps_reg_01"] * (data["ps_car_14"] * (-(data["loo_ps_ind_02_cat"]))))) - data["loo_ps_ind_02_cat"]))
    v["328"] = 0.019992*np.tanh((data["loo_ps_ind_04_cat"] * ((data["ps_ind_03"] + data["loo_ps_ind_04_cat"]) * (-(data["loo_ps_car_08_cat"])))))
    v["329"] = 0.020000*np.tanh((data["ps_car_11"] * ((data["loo_ps_ind_02_cat"] + ((data["loo_ps_ind_07_bin"] + (data["loo_ps_car_07_cat"] * data["missing"]))/2.0))/2.0)))
    v["330"] = 0.019996*np.tanh((((data["loo_ps_car_02_cat"] - (data["loo_ps_ind_06_bin"] - data["loo_ps_car_02_cat"])) * data["loo_ps_car_09_cat"]) * data["loo_ps_ind_02_cat"]))
    v["331"] = 0.018343*np.tanh((((data["loo_ps_car_07_cat"] + (data["ps_reg_03"] + np.tanh(data["loo_ps_car_07_cat"])))/2.0) * (-(data["missing"]))))
    v["332"] = 0.019750*np.tanh((-((data["loo_ps_ind_02_cat"] * (data["loo_ps_ind_18_bin"] * (data["ps_car_14"] * data["loo_ps_car_04_cat"]))))))
    v["333"] = 0.019949*np.tanh(((data["loo_ps_ind_05_cat"] * (-(data["loo_ps_car_06_cat"]))) * (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_04_cat"])))
    v["334"] = 0.020000*np.tanh((data["loo_ps_car_06_cat"] * ((2.800000 + data["ps_car_14"]) * (data["loo_ps_ind_13_bin"] * data["loo_ps_car_04_cat"]))))
    v["335"] = 0.020000*np.tanh(np.tanh((data["ps_ind_03"] * np.tanh(np.tanh((data["ps_reg_02"] * (-(data["ps_ind_15"]))))))))
    v["336"] = 0.020000*np.tanh((data["loo_ps_car_10_cat"] * ((data["loo_ps_ind_11_bin"] + data["loo_ps_ind_18_bin"]) * (data["ps_reg_02"] + data["ps_reg_02"]))))
    v["337"] = 0.020000*np.tanh(np.tanh(np.tanh((data["ps_ind_03"] * ((data["ps_ind_03"] + data["ps_car_11"]) + data["ps_car_11"])))))
    v["338"] = 0.019984*np.tanh(((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_11_bin"] - (data["ps_car_11"] * data["loo_ps_ind_06_bin"]))) * data["loo_ps_car_01_cat"]))
    v["339"] = 0.020000*np.tanh((data["ps_car_15"] * (data["loo_ps_ind_12_bin"] * (data["loo_ps_car_08_cat"] + data["loo_ps_ind_07_bin"]))))
    v["340"] = 0.019953*np.tanh((data["ps_ind_15"] * (data["loo_ps_ind_02_cat"] * (0.600000 - (data["loo_ps_ind_10_bin"] - data["ps_reg_03"])))))
    v["341"] = 0.017914*np.tanh(((data["loo_ps_car_03_cat"] - (data["ps_car_13"] + data["loo_ps_ind_18_bin"])) * (data["loo_ps_car_03_cat"] * 0.273684)))
    v["342"] = 0.020000*np.tanh((data["loo_ps_car_10_cat"] * (-3.0 + (data["ps_car_14"] * (data["ps_car_15"] - data["loo_ps_car_04_cat"])))))
    v["343"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] * (((data["ps_car_11"] * data["loo_ps_car_09_cat"]) + (data["loo_ps_car_02_cat"] - data["loo_ps_ind_04_cat"]))/2.0)))
    v["344"] = 0.019973*np.tanh(((data["ps_car_13"] * (data["ps_ind_14"] - data["ps_reg_02"])) * data["ps_ind_15"]))
    v["345"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] * ((0.965909 - data["ps_reg_03"]) - data["ps_reg_03"])) * data["loo_ps_ind_08_bin"]))
    v["346"] = 0.019973*np.tanh(((data["ps_ind_01"] * (np.tanh(data["loo_ps_car_02_cat"]) + data["loo_ps_ind_11_bin"])) * (-(data["ps_car_15"]))))
    v["347"] = 0.020000*np.tanh((data["loo_ps_ind_09_bin"] * (((-(data["loo_ps_car_08_cat"])) * data["loo_ps_car_03_cat"]) * data["ps_ind_03"])))
    v["348"] = 0.018535*np.tanh((data["loo_ps_ind_04_cat"] * (data["loo_ps_car_04_cat"] * (data["ps_ind_15"] - (-(data["loo_ps_car_01_cat"]))))))
    v["349"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] * ((((data["loo_ps_ind_10_bin"] + data["loo_ps_car_08_cat"])/2.0) + (-(data["loo_ps_ind_12_bin"])))/2.0)))
    v["350"] = 0.019996*np.tanh(((data["loo_ps_car_10_cat"] * (data["loo_ps_ind_13_bin"] - data["ps_reg_02"])) * (data["loo_ps_ind_17_bin"] + data["ps_ind_03"])))
    v["351"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] * (data["ps_reg_01"] * ((data["missing"] + (data["loo_ps_car_07_cat"] - data["loo_ps_ind_05_cat"]))/2.0))))
    v["352"] = 0.020000*np.tanh((data["ps_car_12"] * (data["ps_car_12"] * ((-(np.tanh(data["loo_ps_car_07_cat"]))) * data["ps_reg_02"]))))
    v["353"] = 0.020000*np.tanh((((-((data["ps_ind_01"] * (data["ps_ind_03"] * data["loo_ps_ind_02_cat"])))) + np.tanh(data["loo_ps_car_08_cat"]))/2.0))
    v["354"] = 0.016128*np.tanh((data["ps_ind_15"] * (data["loo_ps_car_11_cat"] * (((data["ps_ind_15"] * data["loo_ps_car_11_cat"]) + 1.871790)/2.0))))
    v["355"] = 0.019961*np.tanh((data["loo_ps_ind_06_bin"] * (data["loo_ps_ind_12_bin"] * (data["loo_ps_ind_11_bin"] + ((data["ps_ind_15"] + data["loo_ps_car_11_cat"])/2.0)))))
    v["356"] = 0.019836*np.tanh(((np.tanh(data["ps_ind_15"]) + (data["loo_ps_ind_02_cat"] * (data["ps_ind_15"] * (-(data["loo_ps_ind_02_cat"])))))/2.0))
    v["357"] = 0.019840*np.tanh(((data["ps_car_15"] * (data["loo_ps_ind_17_bin"] - 0.093750)) * data["ps_car_15"]))
    v["358"] = 0.020000*np.tanh((-((data["loo_ps_car_09_cat"] * ((data["loo_ps_car_01_cat"] + (data["ps_reg_02"] * data["loo_ps_car_09_cat"]))/2.0)))))
    v["359"] = 0.020000*np.tanh((-(((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_17_bin"]) * ((np.tanh(data["ps_ind_01"]) + 0.452381)/2.0)))))
    v["360"] = 0.020000*np.tanh((data["loo_ps_ind_08_bin"] * (data["ps_car_14"] * (data["loo_ps_ind_02_cat"] - np.tanh(data["loo_ps_car_04_cat"])))))
    v["361"] = 0.019996*np.tanh((((data["loo_ps_ind_11_bin"] + (data["loo_ps_ind_08_bin"] * (data["ps_ind_01"] * data["ps_car_11"])))/2.0) * data["ps_car_11"]))
    v["362"] = 0.020000*np.tanh(((8.47791767120361328) * (data["loo_ps_ind_10_bin"] * (data["ps_car_11"] + (data["ps_car_14"] * data["loo_ps_car_08_cat"])))))
    v["363"] = 0.019152*np.tanh((((data["loo_ps_car_08_cat"] + (data["loo_ps_car_08_cat"] * (-(data["ps_ind_03"]))))/2.0) * data["ps_ind_03"]))
    v["364"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * (data["ps_ind_14"] * ((data["ps_car_12"] + (data["ps_reg_03"] * data["loo_ps_car_01_cat"]))/2.0))))
    v["365"] = 0.019996*np.tanh((((data["ps_car_15"] * (data["ps_ind_01"] * data["ps_reg_02"])) + (data["loo_ps_ind_02_cat"] * data["ps_reg_02"]))/2.0))
    v["366"] = 0.020000*np.tanh((data["loo_ps_ind_12_bin"] * ((data["loo_ps_ind_11_bin"] - data["loo_ps_car_10_cat"]) - (data["loo_ps_ind_17_bin"] * data["loo_ps_ind_07_bin"]))))
    v["367"] = 0.019996*np.tanh((-(((data["loo_ps_ind_17_bin"] * data["loo_ps_car_02_cat"]) * (0.347826 - data["ps_car_15"])))))
    v["368"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * ((data["ps_reg_03"] * (data["loo_ps_car_04_cat"] * data["loo_ps_car_01_cat"])) * data["loo_ps_car_08_cat"])))
    v["369"] = 0.020000*np.tanh((((data["ps_reg_02"] + data["ps_ind_15"]) * data["loo_ps_ind_12_bin"]) * (data["ps_ind_03"] + data["ps_ind_03"])))
    v["370"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * (data["ps_car_12"] * (data["loo_ps_ind_02_cat"] * (0.485294 - data["loo_ps_car_08_cat"])))))
    v["371"] = 0.020000*np.tanh(((data["loo_ps_ind_12_bin"] * (-(data["ps_reg_01"]))) - (data["loo_ps_ind_04_cat"] * np.tanh(data["ps_ind_15"]))))
    v["372"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] * (((data["loo_ps_car_02_cat"] * data["loo_ps_ind_04_cat"]) + data["loo_ps_ind_10_bin"])/2.0)) * data["loo_ps_car_03_cat"]))
    v["373"] = 0.020000*np.tanh(((data["ps_ind_14"] * data["loo_ps_car_11_cat"]) * np.tanh((data["ps_ind_03"] + data["ps_ind_15"]))))
    v["374"] = 0.019984*np.tanh((data["ps_car_12"] * ((data["loo_ps_ind_12_bin"] * (1.526320 - data["ps_car_13"])) + data["loo_ps_ind_12_bin"])))
    v["375"] = 0.019684*np.tanh((data["loo_ps_ind_08_bin"] * ((((data["loo_ps_ind_16_bin"] + np.tanh(data["ps_reg_02"]))/2.0) + 0.100000)/2.0)))
    v["376"] = 0.018172*np.tanh((((data["ps_car_11"] + 1.135800)/2.0) * (data["ps_reg_03"] * (data["ps_ind_03"] + data["loo_ps_ind_12_bin"]))))
    v["377"] = 0.020000*np.tanh(((data["ps_car_11"] * ((data["loo_ps_ind_10_bin"] + data["ps_car_11"]) + 1.526320)) * data["loo_ps_car_10_cat"]))
    v["378"] = 0.020000*np.tanh((data["loo_ps_ind_12_bin"] * ((data["ps_ind_03"] + ((data["ps_ind_15"] + (-(data["ps_car_14"])))/2.0))/2.0)))
    v["379"] = 0.020000*np.tanh((data["ps_car_11"] * (((data["loo_ps_car_09_cat"] + (data["ps_ind_01"] * data["loo_ps_ind_11_bin"]))/2.0) * data["ps_ind_01"])))
    v["380"] = 0.019898*np.tanh((data["loo_ps_ind_12_bin"] * (data["ps_ind_03"] - (data["ps_car_11"] + ((-2.0 + data["loo_ps_car_08_cat"])/2.0)))))
    v["381"] = 0.020000*np.tanh((((data["ps_car_11"] * data["loo_ps_ind_18_bin"]) * data["ps_car_11"]) * data["ps_ind_14"]))
    v["382"] = 0.020000*np.tanh((data["loo_ps_ind_04_cat"] * (data["ps_ind_03"] * ((data["ps_ind_03"] * data["ps_car_12"]) - data["ps_ind_03"]))))
    v["383"] = 0.019980*np.tanh((data["loo_ps_ind_02_cat"] * (((data["ps_car_11"] + data["ps_car_15"])/2.0) + (data["ps_car_15"] * data["ps_car_11"]))))
    v["384"] = 0.019836*np.tanh(((data["loo_ps_car_06_cat"] * data["loo_ps_ind_04_cat"]) * ((data["loo_ps_ind_04_cat"] * data["ps_ind_14"]) + data["loo_ps_car_08_cat"])))
    v["385"] = 0.019965*np.tanh(((data["ps_car_13"] + data["ps_ind_15"]) * (data["ps_ind_03"] * (data["ps_reg_02"] + data["ps_car_15"]))))
    v["386"] = 0.019508*np.tanh((data["ps_ind_15"] * (data["ps_reg_01"] * ((data["loo_ps_car_06_cat"] * data["loo_ps_car_04_cat"]) + data["loo_ps_ind_13_bin"]))))
    v["387"] = 0.015862*np.tanh((((data["ps_reg_02"] + data["loo_ps_car_08_cat"])/2.0) * (data["ps_car_13"] * (data["ps_car_12"] - 2.0))))
    v["388"] = 0.020000*np.tanh((data["ps_car_13"] * (0.100000 - ((data["loo_ps_car_10_cat"] * data["ps_ind_01"]) * data["ps_ind_01"]))))
    v["389"] = 0.019992*np.tanh((data["ps_car_13"] * (data["loo_ps_ind_12_bin"] * (data["ps_reg_01"] + data["ps_ind_15"]))))
    v["390"] = 0.015315*np.tanh(((np.tanh(np.tanh((data["loo_ps_ind_07_bin"] + data["loo_ps_car_06_cat"]))) + np.tanh(data["loo_ps_car_08_cat"]))/2.0))
    v["391"] = 0.020000*np.tanh((data["ps_car_15"] * (0.452381 - (data["loo_ps_car_08_cat"] * ((data["loo_ps_ind_11_bin"] + data["ps_car_15"])/2.0)))))
    v["392"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_06_bin"]) * ((data["loo_ps_car_10_cat"] - data["loo_ps_car_02_cat"]) * data["loo_ps_car_10_cat"])))
    v["393"] = 0.015452*np.tanh((data["loo_ps_ind_06_bin"] * ((data["loo_ps_car_08_cat"] + (data["loo_ps_ind_09_bin"] * data["loo_ps_car_09_cat"]))/2.0)))
    v["394"] = 0.019977*np.tanh((data["loo_ps_ind_17_bin"] * ((data["loo_ps_ind_08_bin"] * data["ps_car_15"]) * (0.485294 - data["loo_ps_ind_02_cat"]))))
    v["395"] = 0.019977*np.tanh((data["loo_ps_ind_04_cat"] * ((data["ps_reg_02"] + ((data["loo_ps_ind_04_cat"] - data["ps_ind_01"]) - data["loo_ps_car_09_cat"]))/2.0)))
    v["396"] = 0.019980*np.tanh((data["ps_reg_01"] * (data["loo_ps_car_10_cat"] * (-((data["loo_ps_ind_12_bin"] + data["ps_reg_02"]))))))
    v["397"] = 0.020000*np.tanh(((data["ps_car_15"] * ((data["loo_ps_ind_16_bin"] + data["loo_ps_car_10_cat"])/2.0)) * (data["ps_car_15"] - data["loo_ps_ind_16_bin"])))
    v["398"] = 0.020000*np.tanh((((-((data["loo_ps_ind_02_cat"] * (data["ps_car_14"] * data["loo_ps_ind_07_bin"])))) + np.tanh(data["ps_car_15"]))/2.0))
    v["399"] = 0.014948*np.tanh((data["ps_reg_02"] * (data["loo_ps_car_08_cat"] * (-(data["ps_ind_03"])))))
    v["400"] = 0.016214*np.tanh((data["loo_ps_car_04_cat"] * np.tanh((data["loo_ps_car_11_cat"] - (data["loo_ps_ind_07_bin"] + 1.526320)))))
    return Outputs(v.sum(axis=1))


def GPAri(data):
    return (GPI(data)+GPII(data))/2.


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


def GetData(strdirectory):
    # Project Categorical inputs to Target
    highcardinality = ['ps_car_02_cat',
                       'ps_car_09_cat',
                       'ps_ind_04_cat',
                       'ps_ind_05_cat',
                       'ps_car_03_cat',
                       'ps_ind_08_bin',
                       'ps_car_05_cat',
                       'ps_car_08_cat',
                       'ps_ind_06_bin',
                       'ps_ind_07_bin',
                       'ps_ind_12_bin',
                       'ps_ind_18_bin',
                       'ps_ind_17_bin',
                       'ps_car_07_cat',
                       'ps_car_11_cat',
                       'ps_ind_09_bin',
                       'ps_car_10_cat',
                       'ps_car_04_cat',
                       'ps_car_01_cat',
                       'ps_ind_02_cat',
                       'ps_ind_10_bin',
                       'ps_ind_11_bin',
                       'ps_car_06_cat',
                       'ps_ind_13_bin',
                       'ps_ind_16_bin']

    train = pd.read_csv(strdirectory+'train.csv')
    test = pd.read_csv(strdirectory+'test.csv')

    train['missing'] = (train==-1).sum(axis=1).astype(float)
    test['missing'] = (test==-1).sum(axis=1).astype(float)

    unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
    train.drop(unwanted,inplace=True,axis=1)
    test.drop(unwanted,inplace=True,axis=1)

    test['target'] = np.nan
    feats = list(set(train.columns).difference(set(['id','target'])))
    feats = list(['id'])+feats +list(['target'])
    train = train[feats]
    test = test[feats]
    
    blindloodata = None
    folds = 5
    kf = StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
    for i, (train_index, test_index) in enumerate(kf.split(range(train.shape[0]),train.target)):
        print('Fold:',i)
        blindtrain = train.loc[test_index].copy() 
        vistrain = train.loc[train_index].copy()

        for c in highcardinality:
            blindtrain.insert(1,'loo_'+c, ProjectOnMean(vistrain,
                                                       blindtrain,c))
        if(blindloodata is None):
            blindloodata = blindtrain.copy()
        else:
            blindloodata = pd.concat([blindloodata,blindtrain])

    for c in highcardinality:
        test.insert(1,'loo_'+c, ProjectOnMean(train,
                                             test,c))
    test.drop(highcardinality,inplace=True,axis=1)

    train = blindloodata
    train.drop(highcardinality,inplace=True,axis=1)
    train = train.fillna(train.mean())
    test = test.fillna(train.mean())

    print('Scale values')
    ss = StandardScaler()
    features = train.columns[1:-1]
    ss.fit(pd.concat([train[features],test[features]]))
    train[features] = ss.transform(train[features] )
    test[features] = ss.transform(test[features] )
    train[features] = np.round(train[features], 6)
    test[features] = np.round(test[features], 6)
    return train, test
    

def main():
    print('Started')
    strdirectory = 'D:/Downloads/porto_seguro/my_model/'
    gptrain, gptest = GetData(strdirectory)
    print('GPAri Gini Score:', GiniScore(gptrain.target,GPAri(gptrain)))
    basic = pd.read_csv(strdirectory+'sample_submission.csv')
    basic.target = GPAri(gptest).ravel()
    basic.to_csv(strdirectory+'test_gpari.csv',index=None,float_format='%.6f')
    
    basic = pd.read_csv(strdirectory+'train.csv')
    basic.target = GPAri(gptrain).ravel()
    basic[['target']].to_csv(strdirectory+'train_gpari.csv',index=None,float_format='%.6f')
    print('Finished')


if __name__ == "__main__":
    main()


# # tensoflow  https://www.kaggle.com/camnugent/deep-neural-network-insurance-claims-0-268

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

test_dat = pd.read_csv(base_path + 'test.csv')
train_dat = pd.read_csv(base_path + 'train.csv')
submission = pd.read_csv(base_path + 'sample_submission.csv')

train_y = train_dat['target']
train_x = train_dat.drop(['target', 'id'], axis = 1)
test_dat = test_dat.drop(['id'], axis = 1)

merged_dat = pd.concat([train_x, test_dat],axis=0)

#change data to float32
for c, dtype in zip(merged_dat.columns, merged_dat.dtypes): 
    if dtype == np.float64:     
        merged_dat[c] = merged_dat[c].astype(np.float32)

#one hot encode the categoricals
cat_features = [col for col in merged_dat.columns if col.endswith('cat')]
for column in cat_features:
    temp=pd.get_dummies(pd.Series(merged_dat[column]))
    merged_dat=pd.concat([merged_dat,temp],axis=1)
    merged_dat=merged_dat.drop([column],axis=1)

#standardize the scale of the numericals
numeric_features = [col for col in merged_dat.columns if '_calc_' in  str(col)]
numeric_features = [col for col in numeric_features if '_bin' not in str(col)]

scaler = StandardScaler()
scaled_numerics = scaler.fit_transform(merged_dat[numeric_features])
scaled_num_df = pd.DataFrame(scaled_numerics, columns =numeric_features )


merged_dat = merged_dat.drop(numeric_features, axis=1)

merged_dat = np.concatenate((merged_dat.values,scaled_num_df), axis = 1)

train_x = merged_dat[:train_x.shape[0]]
test_dat = merged_dat[train_x.shape[0]:]


config = tf.contrib.learn.RunConfig(tf_random_seed=42)

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(train_x)

dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[150,150,150], n_classes=2,
                                         feature_columns=feature_cols, config=config)

dnn_clf.fit(train_x, train_y, batch_size=50, steps=40000)

dnn_y_pred = dnn_clf.predict_proba(test_dat)
dnn_y_pred_train = dnn_clf.predict_proba(train_x)


dnn_out = list(dnn_y_pred)

dnn_output = submission
dnn_output['target'] = [x[1] for x in dnn_out]


dnn_output.to_csv(base_path + 'test_dnn_predictions.csv', index=False, float_format='%.4f')

train = pd.read_csv(base_path + 'train.csv')
sub = pd.DataFrame()
sub['id'] = train['id']
sub['target'] = [x[1] for x in list(dnn_y_pred_train)]
sub.to_csv(base_path + 'train_dnn_predictions.csv', index=False, float_format='%.4f')


# # kinetics https://www.kaggle.com/alexandrudaia/kinetic-and-transforms-0-482-up-the-board

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# 
train=pd.read_csv(base_path + 'train_p.csv')
test=pd.read_csv(base_path + 'test_p.csv')
#more about kinetic features  developed  by Daia Alexandru    here  on the next  blog  please  read  last article :
#https://alexandrudaia.quora.com/

##############################################creatinng   kinetic features for  train #####################################################
def  kinetic(row):
    probs=np.unique(row,return_counts=True)[1]/len(row)
    kinetic=np.sum(probs**2)
    return kinetic
    

first_kin_names=[col for  col in train.columns  if '_ind_' in col]
subset_ind=train[first_kin_names]
kinetic_1=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_1.append(k)
second_kin_names= [col for  col in train.columns  if '_car_' in col and col.endswith('cat')]
subset_ind=train[second_kin_names]
kinetic_2=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_2.append(k)
third_kin_names= [col for  col in train.columns  if '_calc_' in col and  not col.endswith('bin')]
subset_ind=train[second_kin_names]
kinetic_3=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_3.append(k)
fd_kin_names= [col for  col in train.columns  if '_calc_' in col and  col.endswith('bin')]
subset_ind=train[fd_kin_names]
kinetic_4=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_4.append(k)
train['kinetic_1']=np.array(kinetic_1)
train['kinetic_2']=np.array(kinetic_2)
train['kinetic_3']=np.array(kinetic_3)
train['kinetic_4']=np.array(kinetic_4)

############################################reatinng   kinetic features for  test###############################################################

first_kin_names=[col for  col in test.columns  if '_ind_' in col]
subset_ind=test[first_kin_names]
kinetic_1=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_1.append(k)
second_kin_names= [col for  col in test.columns  if '_car_' in col and col.endswith('cat')]
subset_ind=test[second_kin_names]
kinetic_2=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_2.append(k)
third_kin_names= [col for  col in test.columns  if '_calc_' in col and  not col.endswith('bin')]
subset_ind=test[second_kin_names]
kinetic_3=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_3.append(k)
fd_kin_names= [col for  col in test.columns  if '_calc_' in col and  col.endswith('bin')]
subset_ind=test[fd_kin_names]
kinetic_4=[]
for row in range(subset_ind.shape[0]):
    row=subset_ind.iloc[row]
    k=kinetic(row)
    kinetic_4.append(k)
test['kinetic_1']=np.array(kinetic_1)
test['kinetic_2']=np.array(kinetic_2)
test['kinetic_3']=np.array(kinetic_3)
test['kinetic_4']=np.array(kinetic_4)

##################################################################end  of kinetics ############################################################################
from sklearn import *
import xgboost as xgb
from multiprocessing import *
import numpy as np
import pandas as pd
from sklearn import *
import xgboost as xgb
import lightgbm as lgb
from multiprocessing import *


d_median = train.median(axis=0)
d_mean = train.mean(axis=0)

def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c:
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(int)
    return df

def multi_transform(df):
    print('Init Shape: ', df.shape)
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    print('After Shape: ', df.shape)
    return df

def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)

def gini_xgb(pred, y):
    #y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)

params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 'objective': 'binary:logistic', 'eval_metric': 'auc', 'seed': 99, 'silent': True}
x1, x2, y1, y2 = model_selection.train_test_split(train, train['target'], test_size=0.25, random_state=99)



# In[ ]:


#keep dist
x1 = transform_df(x1)
y1 = x1['target']
x2 = transform_df(x2)
y2 = x2['target']
test = transform_df(test)

col = [c for c in x1.columns if c not in ['id','target']]
x1 = x1[col]
x2 = x2[col]

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 5000,  watchlist, verbose_eval=50, early_stopping_rounds=100)
print(gini_xgb(model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), y2))
test['target'] = model.predict(xgb.DMatrix(test[col]), ntree_limit=model.best_ntree_limit)


# In[ ]:


test[['id','target']].to_csv(base_path + 'test_uberKinetics.csv', index=False, float_format='%.5f')


# In[ ]:


train = transform_df(train)


# In[ ]:


train['target'] = model.predict(xgb.DMatrix(train[col]), ntree_limit=model.best_ntree_limit)

train[['id','target']].to_csv(base_path + 'train_uberKinetics.csv', index=False, float_format='%.5f')


# # MY PART (STACKING)

# In[ ]:



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# test files

test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
test_dnn = pd.read_csv(base_path + 'test_dnn_predictions.csv')
test_up = pd.read_csv(base_path + 'test_submission.csv')
test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
test_gp = pd.read_csv(base_path + 'test_gpari.csv')

test=pd.read_csv(base_path + 'test.csv')


test = pd.concat([test, 
                   test_xgb[['target']].rename(columns = {'target' : 'xgb'}),
                   test_lgb[['target']].rename(columns = {'target' : 'lgb'}),
                   test_dnn[['target']].rename(columns = {'target' : 'dnn'}),
                   test_up[['target']].rename(columns = {'target' : 'up'}),
                   test_cat[['target']].rename(columns = {'target' : 'cat'}),
                   test_kin[['target']].rename(columns = {'target' : 'kin'}),
                   test_gp[['target']].rename(columns = {'target' : 'gp'})                   
                  ], axis = 1)


train_cols = ['xgb', 'lgb', 'dnn', 'up', 'cat', 'kin', 'gp']


# In[ ]:


### preprocess


# In[ ]:


for t in train_cols:
    test[t + '_rank'] = test[t].rank()


test['target'] = (test['xgb_rank'] + test['lgb_rank'] + test['dnn_rank'] + test['up_rank'] +                  test['cat_rank'] + test['kin_rank'] + test['gp_rank']) / (7 * test.shape[0])


# # The final submission

# In[ ]:


test[['id', 'target']].to_csv(base_path + 'rank_avg.csv.gz', index = False, compression = 'gzip') 

