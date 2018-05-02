
# coding: utf-8

# There are several things we will be doing here. Be warned:
# 
# ![TL;DR](https://m.popkey.co/3c4432/1Z7Mx.gif)
# 
# 
# First, we will do some feature engineering on "categorical" variables (note that I am legally obligated to put that word in quotation marks since, on the surface, they are all numerical variables). I will advertise [__MLBox__](https://github.com/AxeldeRomblay/MLBox) as it will help us with feature engineering. This seems like an excellent ML package, and even though I would not want a single ML package doing everything while I'm just watching, it is undeniable that there are lots of useful tools in it. The one we will use is its [__categorical encoder__](http://mlbox.readthedocs.io/en/latest/features.html#categorical-features). Originally, I wrote this script with [__entity embedding__](https://arxiv.org/abs/1604.06737) as my strategy of choice. We all know what happens with best laid plans ...
# 
# 
# ![Best laid plans](https://i.imgur.com/f8sAmnn.gif)
# 
# 
# On my GTX 1080 the entity embedding learning took 3 minutes, while on Kaggle it was going for solid  52 minutes during peak hours. So I went with [__random projection__](https://en.wikipedia.org/wiki/Random_projection) instead for the sake of time, but I do encourage you to uncomment the line below that calls entity embedding and give it a try locally.
# 
# Next, we will use these new features as an input for [__XGBoost upsampling__](https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283). That script is very fast so it stands a chance of finishing several runs in an hour, and I like the idea as well. I have left all of the original comments from that script intact, which also give credit to other Kagglers from whom @[olivier](https://www.kaggle.com/ogrellier) has borrowed.
# 
# Please read the comment section in that script and @olivier's though on a variety of topics, including the potential for overfitting. Though we are not using his target encoding method here, the same disclaimer applies.
# 
# The idea is to do several quick Bayesian optimization runs with relatively high learing rate (0.1) in order to find the best parameters. Once we have the parameters, proper XGBoost training and prediction are done for higher number of iterations and with lower learning rate (0.02). You can explore other Bayesian optimization ideas [__here__](https://www.kaggle.com/tilii7/bayesian-optimization-of-xgboost-parameters).

# In[ ]:


# coding: utf-8
# The next line is needed for python 2.7 ; probably not for python 3
from __future__ import print_function

import numpy as np
import pandas as pd
import gc
import warnings
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
import xgboost as xgb
from xgboost import XGBClassifier
import gc
from numba import jit
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime
from mlbox.encoding import Categorical_encoder as CE

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

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


# Here we define cross-validation variables that are used for parameter search. Each parameter has its own line, so it is easy to comment something out if you wish. Keep in mind that in such a case you must comment out the matching lines in optimization and explore sections below. I commented out *max_delta_step, subsample and colsample_bytree* and assigned them fixed values. This was done after noticing interesting patterns for alpha, lambda and scale_pos_weight in [__this script__](https://www.kaggle.com/aharless/xgboost-cv-lb-284). So I included them in optimization even though I believe that the above-mentioned script is over-fitting. Feel free to uncomment the lines and optimize 9 instead of 6 variables, but keep in mind that you will need much larger number of initial and optimization points to do that properly.
# 
# Note that the learning rate ("eta") is set to 0.1 below. That is done so we can learn the parameters quickly (without going over 200 XGBoost iterations on average). __Here is a tip: change n_estimators below from 200 to 300-400 and see if that gives a better score during optimization -- it will take longer, though.__

# In[ ]:


# Comment out any parameter you don't want to test
def XGB_CV(
          max_depth,
          gamma,
          min_child_weight,
#          max_delta_step,
#          subsample,
#          colsample_bytree
          scale_pos_weight,
          reg_alpha,
          reg_lambda
         ):

    global GINIbest

    n_splits = 5
    n_estimators = 200
    folds = StratifiedKFold(n_splits=n_splits, random_state=1001)
    xgb_evals = np.zeros((n_estimators, n_splits))
    oof = np.empty(len(trn_df))
    sub_preds = np.zeros(len(sub_df))
    increase = True
    np.random.seed(0)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
        trn_dat, trn_tgt = trn_df.iloc[trn_idx], target.iloc[trn_idx]
        val_dat, val_tgt = trn_df.iloc[val_idx], target.iloc[val_idx]

#
# Define all XGboost parameters
#
        clf = XGBClassifier(n_estimators=n_estimators,
                            max_depth=int(max_depth),
                            objective="binary:logistic",
                            learning_rate=0.1,
#                            subsample=max(min(subsample, 1), 0),
#                            colsample_bytree=max(min(colsample_bytree, 1), 0),
#                            max_delta_step=int(max_delta_step),
                            max_delta_step=1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            gamma=gamma,
                            reg_alpha=reg_alpha,
                            reg_lambda=reg_lambda,
                            scale_pos_weight=scale_pos_weight,
                            min_child_weight=min_child_weight,
                            nthread=4)

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

        clf.fit(trn_dat, trn_tgt,
                eval_set=[(trn_dat, trn_tgt), (val_dat, val_tgt)],
                eval_metric=gini_xgb,
                early_stopping_rounds=None,
                verbose=False)

        # Find best round for validation set
        xgb_evals[:, fold_] = clf.evals_result_["validation_1"]["gini"]
        # Xgboost provides best round starting from 0 so it has to be incremented
        best_round = np.argsort(xgb_evals[:, fold_])[::-1][0]

    # Compute mean score and std
    mean_eval = np.mean(xgb_evals, axis=1)
    std_eval = np.std(xgb_evals, axis=1)
    best_round = np.argsort(mean_eval)[::-1][0]

    print(' Stopped after %d iterations with val-gini = %.6f +- %.6f' % ( best_round, mean_eval[best_round], std_eval[best_round]) )
    if ( mean_eval[best_round] > GINIbest ):
        GINIbest = mean_eval[best_round]

    return mean_eval[best_round]


# I explained above why I went with random projection over entity embedding, but I encourage you to give the latter a try. I suggest you save the files with learned embeddings, so next time you just open them and skip the learning part.
# 
# We are dropping all __ps_calc__ variables.

# In[ ]:


GINIbest = -1.

#ce = CE(strategy='random_projection', verbose=True)
ce = CE(strategy='entity_embedding', verbose=True)

start_time = timer(None)

train_loader = pd.read_csv('../input/train.csv', dtype={'target': np.int8, 'id': np.int32})
train = train_loader.drop(['target', 'id'], axis=1)
print('\n Shape of raw train data:', train.shape)
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train.drop(col_to_drop, axis=1, inplace=True)
target = train_loader['target']
train_ids = train_loader['id'].values

test_loader = pd.read_csv('../input/test.csv', dtype={'id': np.int32})
test = test_loader.drop(['id'], axis=1)
print(' Shape of raw test data:', test.shape)
test.drop(col_to_drop, axis=1, inplace=True)
test_ids = test_loader['id'].values

#n_train = train.shape[0]
#train_test = pd.concat((train, test)).reset_index(drop=True)
col_to_embed = train.columns[train.columns.str.endswith('_cat')].astype(str).tolist()
embed_train = train[col_to_embed].astype(np.str)
embed_test = test[col_to_embed].astype(np.str)
train.drop(col_to_embed, axis=1, inplace=True)
test.drop(col_to_embed, axis=1, inplace=True)

print('\n Learning random projections - this will take less time than entity embedding ...')
#print('\n Learning entity embedding - this will take a while ...')
ce.fit(embed_train, target)
embed_enc_train = ce.transform(embed_train)
embed_enc_test = ce.transform(embed_test)
trn_df = pd.concat((train, embed_enc_train), axis=1)
sub_df = pd.concat((test, embed_enc_test), axis=1)
print('\n Shape of processed train data:', trn_df.shape)
print(' Shape of processed test data:', sub_df.shape)

timer(start_time)


# Several things are worth noting here. First, the effective range of max_depth is 2-6. Since in that range the overfitting is less likely, I was brave enough to top *gamma* and *min_child_weight* at 5. All of this is done for the sake of time. However, a proper way would be to allow max_depth to be 8 (or even 10), in which case *gamma* and *min_child_weight* should be topping at 10 or so.
# 
# *If you decide to uncomment the remaining three parameters here, the same must be done above in XGB_CV section.*

# We are doing a little trick here. Since it is highly unlikely that 5-6 parameter search runs would be able to identify anything remotely close to optimal parameters, I am giving us a head-start by providing two parameter combinations that are known to give good scores.
# 
# Note that these are specifically for random projection encoding. If you go with entity embedding, you'll want to delete this section and uncomment the whole paragraph underneath it.

# In[1]:


XGB_BO = BayesianOptimization(XGB_CV, {
                                     'max_depth': (2, 6.99),
                                     'gamma': (0.1, 5),
                                     'min_child_weight': (0, 5),
                                     'scale_pos_weight': (1, 5),
                                     'reg_alpha': (0, 10),
                                     'reg_lambda': (1, 10),
#                                     'max_delta_step': (0, 5),
#                                     'subsample': (0.4, 1.0),
#                                     'colsample_bytree' :(0.4, 1.0)
                                    })

#XGB_BO.explore({
#              'max_depth':            [4, 4],
#              'gamma':                [0.1511, 2.7823],
#              'min_child_weight':     [2.4073, 2.6086],
#              'scale_pos_weight':     [2.2281, 2.4993],
#              'reg_alpha':            [8.0702, 6.9874],
#              'reg_lambda':           [2.0126, 3.9598],
#              'max_delta_step':       [1, 1],
#              'subsample':            [0.8, 0.8],
#              'colsample_bytree':     [0.8, 0.8],
#              })

# If you go with entitiy embedding, these are good starting points
#XGB_BO.explore({
#              'max_depth':            [4, 4],
#              'gamma':                [2.8098, 2.1727],
#              'min_child_weight':     [4.1592, 4.8113],
#              'scale_pos_weight':     [2.4450, 1.7195],
#              'reg_alpha':            [2.8601, 7.6995],
#              'reg_lambda':           [6.5563, 2.6879],
#              })


# We are doing only one random guess of parameters, which makes a total of 3 when combined with two exploratory groups above. Afterwards, only 2 optimization runs are done.
# 
# A total number of random points (from **.explore** section + init_points) should be at least 10-15. I would consider 20 if you decide to include more than 6 parameters. n_iter should be in 30+ range to do proper parameter optimization.

# In[ ]:


print('-'*126)

start_time = timer(None)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    XGB_BO.maximize(init_points=1, n_iter=2, acq='ei', xi=0.0)
timer(start_time)


# Here we print the summary and create a CSV file with grid results.

# In[ ]:


print('-'*126)
print('\n Final Results')
print(' Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'])
print(' Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'])
grid_file = 'Bayes-gini-5fold-XGB-target-enc-run-04-v1-grid.csv'
print(' Saving grid search parameters to %s' % grid_file)
XGB_BO.points_to_csv(grid_file)


# Finally, we do the last XGBoost upsampling, but this time with larger **n_estimators** and smaller **learning_rate**. You should do 1000 for n_estimators even if you don't touch the learning rate. If you lower the learning rate further, definitely increase n_estimators to 1500-2000.

# In[ ]:



max_depth = int(XGB_BO.res['max']['max_params']['max_depth'])
gamma = XGB_BO.res['max']['max_params']['gamma']
min_child_weight = XGB_BO.res['max']['max_params']['min_child_weight']
#max_delta_step = int(XGB_BO.res['max']['max_params']['max_delta_step'])
#subsample = XGB_BO.res['max']['max_params']['subsample']
#colsample_bytree = XGB_BO.res['max']['max_params']['colsample_bytree']
scale_pos_weight = XGB_BO.res['max']['max_params']['scale_pos_weight']
reg_alpha = XGB_BO.res['max']['max_params']['reg_alpha']
reg_lambda = XGB_BO.res['max']['max_params']['reg_lambda']

start_time = timer(None)
print('\n Making final prediction - this will take a while ...')
n_splits = 5
n_estimators = 800
folds = StratifiedKFold(n_splits=n_splits, random_state=1001)
imp_df = np.zeros((len(trn_df.columns), n_splits))
xgb_evals = np.zeros((n_estimators, n_splits))
oof = np.empty(len(trn_df))
sub_preds = np.zeros(len(sub_df))
increase = True
np.random.seed(0)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(target, target)):
    trn_dat, trn_tgt = trn_df.iloc[trn_idx], target.iloc[trn_idx]
    val_dat, val_tgt = trn_df.iloc[val_idx], target.iloc[val_idx]

    clf = XGBClassifier(n_estimators=n_estimators,
                        max_depth=max_depth,
                        objective="binary:logistic",
                        learning_rate=0.02,
#                        subsample=subsample,
#                        colsample_bytree=colsample_bytree,
#                        max_delta_step=max_delta_step,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        max_delta_step=1,
                        gamma=gamma,
                        min_child_weight=min_child_weight,
                        reg_alpha=reg_alpha,
                        reg_lambda=reg_lambda,
                        scale_pos_weight=scale_pos_weight,
                        nthread=4)
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

    clf.fit(trn_dat, trn_tgt,
            eval_set=[(trn_dat, trn_tgt), (val_dat, val_tgt)],
            eval_metric=gini_xgb,
            early_stopping_rounds=None,
            verbose=False)

    # Keep feature importances
    imp_df[:, fold_] = clf.feature_importances_

    # Find best round for validation set
    xgb_evals[:, fold_] = clf.evals_result_["validation_1"]["gini"]
    # Xgboost provides best round starting from 0 so it has to be incremented
    best_round = np.argsort(xgb_evals[:, fold_])[::-1][0]

    # Predict OOF and submission probas with the best round
    oof[val_idx] = clf.predict_proba(val_dat, ntree_limit=best_round)[:, 1]
    # Update submission
    sub_preds += clf.predict_proba(sub_df, ntree_limit=best_round)[:, 1] / n_splits

    # Display results
    print("Fold %2d : %.6f @%4d / best score is %.6f @%4d"
          % (fold_ + 1,
             eval_gini(val_tgt, oof[val_idx]),
             n_estimators,
             xgb_evals[best_round, fold_],
             best_round))

print("Full OOF score : %.6f" % eval_gini(target, oof))

# Compute mean score and std
mean_eval = np.mean(xgb_evals, axis=1)
std_eval = np.std(xgb_evals, axis=1)
best_round = np.argsort(mean_eval)[::-1][0]

print("Best mean score : %.6f + %.6f @%4d"
      % (mean_eval[best_round], std_eval[best_round], best_round))

best_gini = round(mean_eval[best_round], 6)
importances = sorted([(trn_df.columns[i], imp) for i, imp in enumerate(imp_df.mean(axis=1))],
                     key=lambda x: x[1])

for f, imp in importances[::-1]:
    print("%-34s : %10.4f" % (f, imp))

timer(start_time)

final_df = pd.DataFrame(test_ids, columns=['id'])
final_df['target'] = sub_preds

now = datetime.now()
sub_file = 'submission_5fold-xgb-upsampling-target-enc-01_' + str(best_gini) + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
print('\n Writing submission: %s' % sub_file)
final_df.to_csv(sub_file, index=False, float_format="%.9f")

