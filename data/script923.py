
# coding: utf-8

# Make sure to install the superb [__Bayesian Optimization__](https://github.com/fmfn/BayesianOptimization) library.

# In[ ]:


# This line is needed for python 2.7 ; probably not for python 3
from __future__ import print_function

import numpy as np
import pandas as pd
import gc
import warnings

from bayes_opt import BayesianOptimization

from sklearn.cross_validation import cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import contextlib


# This will be used to capture stderr and stdout without having anything print on screen.
# 
# **It turns out that Kaggle does not have "cStringIO", so I will comment out this portion.**

# In[ ]:


#@contextlib.contextmanager
#def capture():
#    import sys
#    from cStringIO import StringIO
#    olderr, oldout = sys.stderr, sys.stdout
#    try:
#        out=[StringIO(), StringIO()]
#        sys.stderr,sys.stdout = out
#        yield out
#    finally:
#        sys.stderr,sys.stdout = olderr,oldout
#        out[0] = out[0].getvalue().splitlines()
#        out[1] = out[1].getvalue().splitlines()


# Scaling is really not needed for XGBoost, but I leave it here in case if you do the optimization using ML approaches that need it.

# In[ ]:


def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


# Loading files.

# In[ ]:


DATA_TRAIN_PATH = '../input/train.csv'
DATA_TEST_PATH = '../input/test.csv'

def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):
    train_loader = pd.read_csv(path_train, dtype={'target': np.int8, 'id': np.int32})
    train = train_loader.drop(['target', 'id'], axis=1)
    train_labels = train_loader['target'].values
    train_ids = train_loader['id'].values
    print('\n Shape of raw train data:', train.shape)

    test_loader = pd.read_csv(path_test, dtype={'id': np.int32})
    test = test_loader.drop(['id'], axis=1)
    test_ids = test_loader['id'].values
    print(' Shape of raw test data:', test.shape)

    return train, train_labels, test, train_ids, test_ids


# Define cross-validation variables that are used for parameter search. Each parameter has its own line, so it is easy to comment something out if you wish. Keep in mind that in such a case you must comment out the matching lines in optimization and explore sections below.
# 
# *Note that the learning rate ("eta") is set to 0.1 below. That is certainly not optimal, but it will make the search go faster. You will probably want to experiment with values in 0.01-0.05 range, but beware that it will significantly slow down the process because more iterations will be required to get to early stopping. Doing 10-fold instead of 5-fold cross-validation will also result in a small gain, but will double the search time.*
# 
# XGBoost outputs lots of interesting info, but it is not very helpful and clutters the screen when doing grid search. So we will run XGboost CV with verbose turned on, but will capture stderr in result[0] and stdout in result[1]. We will extract the relevant info from these variables later, and will print the record of each CV run into a log file.
# 
# AUC will be optimized here. We can go with separately defined gini scorer and use **feval=gini** below but I don't think it makes any difference because AUC and gini are directly correlated.
# 
# **Commenting out the capture so there will be no record of xgb.cv in log file.**

# In[ ]:


# Comment out any parameter you don't want to test
def XGB_CV(
          max_depth,
          gamma,
          min_child_weight,
          max_delta_step,
          subsample,
          colsample_bytree
         ):

    global AUCbest
    global ITERbest

#
# Define all XGboost parameters
#

    paramt = {
              'booster' : 'gbtree',
              'max_depth' : int(max_depth),
              'gamma' : gamma,
              'eta' : 0.1,
              'objective' : 'binary:logistic',
              'nthread' : 4,
              'silent' : True,
              'eval_metric': 'auc',
              'subsample' : max(min(subsample, 1), 0),
              'colsample_bytree' : max(min(colsample_bytree, 1), 0),
              'min_child_weight' : min_child_weight,
              'max_delta_step' : int(max_delta_step),
              'seed' : 1001
              }

    folds = 5
    cv_score = 0

    print("\n Search parameters (%d-fold validation):\n %s" % (folds, paramt), file=log_file )
    log_file.flush()

    xgbc = xgb.cv(
                    paramt,
                    dtrain,
                    num_boost_round = 20000,
                    stratified = True,
                    nfold = folds,
#                    verbose_eval = 10,
                    early_stopping_rounds = 100,
                    metrics = 'auc',
                    show_stdv = True
               )

# This line would have been on top of this section
#    with capture() as result:

# After xgb.cv is done, this section puts its output into log file. Train and validation scores 
# are also extracted in this section. Note the "diff" part in the printout below, which is the 
# difference between the two scores. Large diff values may indicate that a particular set of 
# parameters is overfitting, especially if you check the CV portion of it in the log file and find 
# out that train scores were improving much faster than validation scores.

#    print('', file=log_file)
#    for line in result[1]:
#        print(line, file=log_file)
#    log_file.flush()

    val_score = xgbc['test-auc-mean'].iloc[-1]
    train_score = xgbc['train-auc-mean'].iloc[-1]
    print(' Stopped after %d iterations with train-auc = %f val-auc = %f ( diff = %f ) train-gini = %f val-gini = %f' % ( len(xgbc), train_score, val_score, (train_score - val_score), (train_score*2-1),
(val_score*2-1)) )
    if ( val_score > AUCbest ):
        AUCbest = val_score
        ITERbest = len(xgbc)

    return (val_score*2) - 1


# The "real" code starts here.

# In[ ]:


# Define the log file. If you repeat this run, new output will be added to it
log_file = open('Porto-AUC-5fold-XGB-run-01-v1-full.log', 'a')
AUCbest = -1.
ITERbest = 0

# Load data set and target values
train, target, test, tr_ids, te_ids = load_data()
n_train = train.shape[0]
train_test = pd.concat((train, test)).reset_index(drop=True)
col_to_drop = train.columns[train.columns.str.endswith('_cat')]
col_to_dummify = train.columns[train.columns.str.endswith('_cat')].astype(str).tolist()

for col in col_to_dummify:
    dummy = pd.get_dummies(train_test[col].astype('category'))
    columns = dummy.columns.astype(str).tolist()
    columns = [col + '_' + w for w in columns]
    dummy.columns = columns
    train_test = pd.concat((train_test, dummy), axis=1)

train_test.drop(col_to_dummify, axis=1, inplace=True)
train_test_scaled, scaler = scale_data(train_test)
train = train_test_scaled[:n_train, :]
test = train_test_scaled[n_train:, :]
print('\n Shape of processed train data:', train.shape)
print(' Shape of processed test data:', test.shape)

# We really didn't need to load the test data in the first place unless you are planning to make
# a prediction at the end of this run.
# del test
# gc.collect()


# I am doing a stratified split and using only 25% of the data. Obviously, this is done to make sure that this notebook can run to completion on Kaggle. In a production version, you should uncomment the first line in the section below, and comment out or delete everything else.

# In[ ]:


# dtrain = xgb.DMatrix(train, label = target)

sss = StratifiedShuffleSplit(target, random_state=1001, test_size=0.75)
for train_index, test_index in sss:
    break
X_train, y_train = train[train_index], target[train_index]
del train, target
gc.collect()
dtrain = xgb.DMatrix(X_train, label = y_train)


# These are the parameters and their ranges that will be used during optimization. They must match the parameters that are passed above to the XGB_CV function. If you commented out any of them above, you should do the same here. Note that these are pretty wide ranges for most parameters.

# In[ ]:


XGB_BO = BayesianOptimization(XGB_CV, {
                                     'max_depth': (2, 12),
                                     'gamma': (0.001, 10.0),
                                     'min_child_weight': (0, 20),
                                     'max_delta_step': (0, 10),
                                     'subsample': (0.4, 1.0),
                                     'colsample_bytree' :(0.4, 1.0)
                                    })


# This portion of the code is not necessary. You can simply specify that 10-20 random parameter combinations (**init_points** below) be used. However, I like to try couple of high- and low-end values for each parameter as a starting point, and after that fewer random points are needed. Note that a number of options must be the same for each parameter, and they are applied vertically.

# In[ ]:


XGB_BO.explore({
              'max_depth':            [3, 8, 3, 8, 8, 3, 8, 3],
              'gamma':                [0.5, 8, 0.2, 9, 0.5, 8, 0.2, 9],
              'min_child_weight':     [0.2, 0.2, 0.2, 0.2, 12, 12, 12, 12],
              'max_delta_step':       [1, 2, 2, 1, 2, 1, 1, 2],
              'subsample':            [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
              'colsample_bytree':     [0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8],
              })


# In my version of sklearn there are many warning thrown out by the GP portion of this code. This is set to prevent them from showing on screen.
# 
# If you have a special relationship with your computer and want to know everything it is saying back, you'd probably want to remove the two "warnings" lines and slide the XGB_BO line all the way left.
# 
# I am doing only 2 initial points, which along with 8 exploratory points above makes it 10 "random" parameter combinations. I'd say that 15-20 is usually adequate. For n_iter 25-50 is usually enough.
# 
# There are several commented out maximize lines that could be worth exploring. The exact combination of parameters determines **[exploitation vs. exploration](https://github.com/fmfn/BayesianOptimization/blob/master/examples/exploitation%20vs%20exploration.ipynb)**. It is tough to know which would work better without actually trying, though in my hands exploitation with "expected improvement" usually works the best. That's what the XGB_BO.maximize line below is specifying.

# In[ ]:


print('-'*130)
print('-'*130, file=log_file)
log_file.flush()

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    XGB_BO.maximize(init_points=2, n_iter=5, acq='ei', xi=0.0)

# XGB_BO.maximize(init_points=10, n_iter=50, acq='ei', xi=0.0)
# XGB_BO.maximize(init_points=10, n_iter=50, acq='ei', xi=0.01)
# XGB_BO.maximize(init_points=10, n_iter=50, acq='ucb', kappa=10)
# XGB_BO.maximize(init_points=10, n_iter=50, acq='ucb', kappa=1)


# This portions gives the summary and creates a CSV file with results.

# In[ ]:


print('-'*130)
print('Final Results')
print('Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'])
print('Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'])
print('-'*130, file=log_file)
print('Final Result:', file=log_file)
print('Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'], file=log_file)
print('Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'], file=log_file)
log_file.flush()
log_file.close()

history_df = pd.DataFrame(XGB_BO.res['all']['params'])
history_df2 = pd.DataFrame(XGB_BO.res['all']['values'])
history_df = pd.concat((history_df, history_df2), axis=1)
history_df.rename(columns = { 0 : 'gini'}, inplace=True)
history_df['AUC'] = ( history_df['gini'] + 1 ) / 2
history_df.to_csv('Porto-AUC-5fold-XGB-run-01-v1-grid.csv')


# Good luck! Let me know how it works.
