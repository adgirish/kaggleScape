
# coding: utf-8

# Based on the1owl's [kernel](https://www.kaggle.com/the1owl/forza-baseline-lightgbm-example)

# In[ ]:


MAX_ROUNDS = 1200
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.024


# I recommend initially setting <code>MAX_ROUNDS</code> fairly high and using <code>OPTIMIZE_ROUNDS</code> to get an idea of the appropriate number of rounds (which, in my judgment, should be close to the maximum value of <code>best_iteration</code> among all folds, maybe even a bit higher if your model is adequately regularized...or alternatively, you can look at the detailed output from the boosting rounds and choose a value that seems like it would work OK for all folds).  Then I would turn off <code>OPTIMIZE_ROUNDS</code> and set <code>MAX_ROUNDS</code> to the appropraite number of total rounds.  The problem with "early stopping" by choosing the best round for each fold is that it overfits to the validation data.    It's therefore liable not to produce the optimal model for predicting test data, and if it's used to produce validation predictions for stacking/ensembling with other models, it would cause this one to have too much weight in the ensemble.

# In[ ]:


import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from numba import jit
from sklearn import *
import lightgbm as lgb
from multiprocessing import *


# In[ ]:


# Compute gini

# from CPMP's kernel https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
@jit
def eval_gini(y_true, y_prob):
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


# In[ ]:


def transform_df(df):
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 7:
            for val in one_hot[c]:
                df[c+'_oh_' + str(val)] = (df[c].values == val).astype(np.int)
    return df

def multi_transform(df):
    p = Pool(cpu_count())
    df = p.map(transform_df, np.array_split(df, cpu_count()))
    df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    return df

def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = eval_gini(y, preds) / eval_gini(y, y)
    return 'gini', score, True



# In[ ]:


# Read data
train_df = pd.read_csv('../input/train.csv') # .iloc[0:200,:]
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


# Process data
col = [c for c in train_df.columns if c not in ['id','target']]
col = [c for c in col if not c.startswith('ps_calc_')]

id_test = test_df['id'].values
id_train = train_df['id'].values

y = train_df['target']
X = train_df[col]
y_valid_pred = 0*y
X_test = test_df.drop(['id'], axis=1)
y_test_pred = 0


# In[ ]:


# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)


# In[ ]:


# Set up classifier
params = {
    'learning_rate': LEARNING_RATE, 
    'max_depth': 4, 
    'lambda_l1': 16.7,
    'boosting': 'gbdt', 
    'objective': 'binary', 
    'metric': 'auc',
    'feature_fraction': .7,
    'is_training_metric': False, 
    'seed': 99
}


# In[ ]:


# Run CV

for i, (train_index, test_index) in enumerate(kf.split(train_df)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index].copy()
    X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
    test = test_df.copy()[col]
    print( "\nFold ", i)

    # Transform data for this fold
    one_hot = {c: list(X_train[c].unique()) for c in X_train.columns}
    X_train = X_train.replace(-1, np.NaN)  # Get rid of -1 while computing summary stats
    d_median = X_train.median(axis=0)
    d_mean = X_train.mean(axis=0)
    X_train = X_train.fillna(-1)  # Restore -1 for missing values

    X_train = multi_transform(X_train)
    X_valid = multi_transform(X_valid)
    test = multi_transform(test)

    # Run model for this fold
    if OPTIMIZE_ROUNDS:
        fit_model = lgb.train( 
                               params, 
                               lgb.Dataset(X_train, label=y_train), 
                               MAX_ROUNDS, 
                               lgb.Dataset(X_valid, label=y_valid), 
                               verbose_eval=50, 
                               feval=gini_lgb, 
                               early_stopping_rounds=200 
                             )
        print( " Best iteration = ", fit_model.best_iteration )
        pred = fit_model.predict(X_valid, num_iteration=fit_model.best_iteration)
        test_pred = fit_model.predict(test[col], num_iteration=fit_model.best_iteration)
    else:
        fit_model = lgb.train( 
                               params, 
                               lgb.Dataset(X_train, label=y_train), 
                               MAX_ROUNDS, 
                               verbose_eval=50 
                             )
        pred = fit_model.predict(X_valid)
        test_pred = fit_model.predict(test)

    # Save validation predictions for this fold
    print( "  Gini = ", eval_gini(y_valid, pred) )
    y_valid_pred.iloc[test_index] = (np.exp(pred) - 1.0).clip(0,1)
    
    # Accumulate test set predictions
    y_test_pred += (np.exp(test_pred) - 1.0).clip(0,1)
    
y_test_pred /= K  # Average test set predictions

print( "\nGini for full training set:" )
eval_gini(y, y_valid_pred)


# In[ ]:


# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('lgb_valid.csv', float_format='%.6f', index=False)


# In[ ]:


# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('lgb_submit.csv', float_format='%.6f', index=False)


# version 8:  Resubmitting identical run because version 7 seems to have become invisible<br>
# versions 9,10 (substantively identical): Set <code>lambda_l1=16.7</code>, sorted LB score improved but still reported as .282<br>
# version 11: With <code>OPTIMIZE_ROUNDS</code>, negate gini score so LightGBM will minimize. <code>is_higher_better</code> not working.<br>
# version 12: Un-negate gini score and set <code>is_higher_better</code> again. Maybe I misunderstood.<br>
# version 14: Set <code>MAX_ROUNDS=1400</code> for prediction run.<br>
# versions 15-21: Set <code>feature_fraction</code>. LB score goes up but still reported as .282
