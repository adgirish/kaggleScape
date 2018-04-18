
# coding: utf-8

# # Matthews correlation coefficient
# 
# See https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
# 
# Author: [CPMP](https://www.kaggle.com/cpmpml)
# 
# A fast implementation of [Anokas mcc optimization code](https://www.kaggle.com/c/bosch-production-line-performance/forums/t/22917/optimising-probabilities-binary-prediction-script).
# 
# This code takes as input probabilities, and selects the threshold that yields the best MCC score.  It is efficient enough to be used as a custom evaluation function in xgboost

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import matthews_corrcoef

import matplotlib.pyplot as plt
import numpy as np


# We compile the code with Numba.  If you don't have it installed, then just comment out the import and the @jit lines below.

# In[ ]:


from numba import jit

@jit
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)

@jit
def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc


# Let's see what this gives on an example.  We also compute the MCC using scikit-learn function to make sure we compute the same.

# In[ ]:


y_prob0 = np.random.rand(1000000)
y_prob = y_prob0 + 0.4 * np.random.rand(1000000) - 0.02


y_true = (y_prob0 > 0.6).astype(int)
best_proba, best_mcc, y_pred = eval_mcc(y_true, y_prob, True)


# It is fast.  For a one million items input:

# In[ ]:


get_ipython().run_line_magic('timeit', 'eval_mcc(y_true, y_prob)')


# Probabilities can be identical for several values as pointed out by @Commander.  Let's test it.

# In[ ]:


def roundn(yprob, scale):
    return np.around(y_prob * scale) / scale

best_proba, best_mcc, y_pred = eval_mcc(y_true, roundn(y_prob, 100), True)
            


# We see that the best mcc is lower than when probabilities were all different.

# For use with xgboost, we wrap it to get the right input and output. 

# In[ ]:


def mcc_eval(y_prob, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_prob)
    return 'MCC', best_mcc


# We can then use it with xgboost, for instance passing it as feval parameter as follow:

#     bst = xgb.train(params, Xtrain, 
#                         num_boost_round=num_round, 
#                         evals=watchlist,
#                         early_stopping_rounds=early_stopping_rounds, 
#                         evals_result=evals_result, 
#                         verbose_eval=verbose_eval,
#                         feval=mcc_eval, 
#                         maximize=True,)
