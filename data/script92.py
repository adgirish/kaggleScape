
# coding: utf-8

# ## Preparation
# 
# First we load some libraries...

# In[1]:


import numpy as np
import pandas as pd

from hyperopt import hp, tpe
from hyperopt.fmin import fmin

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer

import xgboost as xgb

import lightgbm as lgbm


# Next we load the data ; note i'm only loading training data because i'm just tuning parameters and comparing models  in this notebook. To speed up tuning, you can also load part of the data (add  nrows=50000), note this will give different results though, so you should re-tune with full data.

# In[2]:


df = pd.read_csv('../input/train.csv')

X = df.drop(['id', 'target'], axis=1)
Y = df['target']


# Next up we define the gini evaluation functions. This is not a standard metric so we need to code it  ourself (actually i borrowed the code for this).

# In[3]:


def gini(truth, predictions):
    g = np.asarray(np.c_[truth, predictions, np.arange(len(truth)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(truth) + 1) / 2.
    return gs / len(truth)

def gini_xgb(predictions, truth):
    truth = truth.get_label()
    return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)

def gini_lgb(truth, predictions):
    score = gini(truth, predictions) / gini(truth, truth)
    return 'gini', score, True

def gini_sklearn(truth, predictions):
    return gini(truth, predictions) / gini(truth, truth)

gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)


# ## Tuning Random Forest
# 
# In this part we'll tune the random forest classsifier, using hyperopt library. The hyperopt library has a similar purpose as gridsearch, but instead of doing an exhaustive search of the parameter space it evaluates a few well-chosen data points and then extrapolates the optimal solution based on modeling. In practice that means it often needs much fewer iterations to find a good solution.
# 
# Random forests work by averaging predictions from many decision trees - the idea is that by averaging many trees the mistakes of each tree are ironed out. Each decision tree can be somewhat overfitted, by averaging them the final result should be good.
# 
# The important parameters to tune are:
# * Number of trees in the forest (n_estimators)
# * Tree complexity (max_depth)

# In[4]:


def objective(params):
    params = {'n_estimators': int(params['n_estimators']), 'max_depth': int(params['max_depth'])}
    clf = RandomForestClassifier(n_jobs=4, class_weight='balanced', **params)
    score = cross_val_score(clf, X, Y, scoring=gini_scorer, cv=StratifiedKFold()).mean()
    print("Gini {:.3f} params {}".format(score, params))
    return score

space = {
    'n_estimators': hp.quniform('n_estimators', 25, 500, 25),
    'max_depth': hp.quniform('max_depth', 1, 10, 1)
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)


# So we can see hyperopt exploring parameter space above - let's see what hyperopt estimates as the optimal parameters?

# In[5]:


print("Hyperopt estimated optimum {}".format(best))


# ## Tuning XGBoost
# 
# Similar to tuning above, now we will tune xgboost parameters using hyperopt!
# 
# XGBoost is also an based on an ensemble of decision trees, but different from random forest. The trees are not averaged, but added. The decision trees are trained to correct residuals from the previous trees. The idea is that many small decision trees are trained, each adding a bit of info to improve overall predictions.
# 
# I will follow this guide for tuning [Tuning XGBoost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)
# 
# I'm initially fixing the number of trees to 250 and learning rate to 0.05 (determined that with a quick experiment) - then we can find good values for the other parameters. Later we can re-iterate this.
# 
# The most important parameters are:
# * Number of trees (n_estimators)
# * Learning rate - later trees have less influence (learning_rate)
# * Tree complexity (max_depth)
# * Gamma - Make individual trees conservative, reduce overfitting 
# * Column sample per tree - reduce overfitting

# In[6]:


def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    
    clf = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        n_jobs=4,
        **params
    )
    
    score = cross_val_score(clf, X, Y, scoring=gini_scorer, cv=StratifiedKFold()).mean()
    print("Gini {:.3f} params {}".format(score, params))
    return score

space = {
    'max_depth': hp.quniform('max_depth', 2, 8, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)


# So hyperopt hass explored the parameter space again - let's have a look at the estimated optimum!

# In[ ]:


print("Hyperopt estimated optimum {}".format(best))


# ## Tuning LightGBM
# 
# LightGBM is very similar to xgboost, it is also uses a gradient boosted tree approach. So the explanation above mostly holds also.
# 
# The important parameters to tune are:
# * Number of estimators
# * Tree complexity - in lightgbm that is controlled by number of leaves (num_leaves)
# * Learning rate
# * Feature fraction
# 
# We will fix number of estimators to 500 and learning rate to 0.01 (chosen experimentally) and tune the remaining parameters with hyperopt. Then later we could revisit for better results! 

# In[ ]:


def objective(params):
    params = {
        'num_leaves': int(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    
    clf = lgbm.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.01,
        **params
    )
    
    score = cross_val_score(clf, X, Y, scoring=gini_scorer, cv=StratifiedKFold()).mean()
    print("Gini {:.3f} params {}".format(score, params))
    return score

space = {
    'num_leaves': hp.quniform('num_leaves', 8, 128, 2),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)


# Now let's see what hyperopt estimates as the optimum?

# In[ ]:


print("Hyperopt estimated optimum {}".format(best))


# ## Intepretation of tuning
# 
# It's interesting to compare the parameters that hyperopt found for random forest and XGBoost. Random forest ended up with 375 trees of depth 7, where XGBoost has 250 of depth 5. This fits the theory that random forest averages many complex (independantly trained) trees to get good results, where xgboost & lightgbm (boosting) add up many simple trees (trained on residuals).

# ## Comparing the models
# 
# Now let's see how the models perform - if hyperopt has determined a sensible set of parameters for us...

# In[ ]:


rf_model = RandomForestClassifier(
    n_jobs=4,
    class_weight='balanced',
    n_estimators=325,
    max_depth=5
)

xgb_model = xgb.XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    n_jobs=4,
    max_depth=2,
    colsample_bytree=0.7,
    gamma=0.15
)

lgbm_model = lgbm.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.01,
    num_leaves=16,
    colsample_bytree=0.7
)

models = [
    ('Random Forest', rf_model),
    ('XGBoost', xgb_model),
    ('LightGBM', lgbm_model),
]

for label, model in models:
    scores = cross_val_score(model, X, Y, cv=StratifiedKFold(), scoring=gini_scorer)
    print("Gini coefficient: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))

