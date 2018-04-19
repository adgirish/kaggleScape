
# coding: utf-8

# ## Introduction
# The goal is this notebook is to see if best rounds computed during 5-CV optimization for boosting algos can be generalized to predicting test target using the full training dataset. 
# 
# The notebook runs 20 2-fold experiments where each fold is used to predict the other by means of a 5-fold CV. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
trn = pd.read_csv("../input/train.csv", index_col=0)
target = trn.target
del trn["target"]


# In[ ]:


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def compute_optimal_round(x, y, x_v, y_v):
    clf = LGBMClassifier(n_estimators=200, 
                         num_leaves=10,
                         learning_rate=.2, n_jobs=2)
    clf.fit(x, y, 
            eval_set=[(x_v, y_v)], 
            early_stopping_rounds=25,
            eval_metric="auc",
            verbose=0)
    best_score = roc_auc_score(y_v, clf.predict_proba(x_v, num_iteration=clf.best_iteration_)[:, 1])
    return clf.best_iteration_, best_score

def compute_score(x, y, x_v, y_v, rounds):
    clf = LGBMClassifier(n_estimators=int(np.max(rounds)), 
                         num_leaves=10,
                         learning_rate=.2, n_jobs=2)
    clf.fit(x, y, 
            eval_set=[(x_v, y_v)], 
            early_stopping_rounds=None,
            eval_metric="auc",
            verbose=0)
    #print(clf.evals_result_)
    lgb_evals = clf.evals_result_["valid_0"]["auc"]
    return [lgb_evals[int(round) - 1] for round in rounds]


# ## Check generalization
# Here we 
# 1. split the dataset in 2 equal parts at each iteration. 
# 2. We run an LGBM on the first part with early stopping using the 2nd part to get the optimal round
# 3. We then run a 5 fold CV with early stopping on the 1st part and keep all folds best round
# 4. We compare mean, max and min to the optimum
# 5. Do the same flipping part 1 and 2 roles
# 6. Use a new seed

# In[ ]:


from sklearn.model_selection import StratifiedKFold
import time
nb_seed = 20
values = np.zeros((2 * nb_seed, 4))
scores = np.zeros((2 * nb_seed, 4))
i = 0
start = time.time()
for seed in range(nb_seed):
    fold_lev1 = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    for trn_l1_idx, val_l1_idx in fold_lev1.split(target, target):
        # Split level1 data
        trn_l1_x, trn_l1_y = trn.iloc[trn_l1_idx], target.iloc[trn_l1_idx]
        val_l1_x, val_l1_y = trn.iloc[val_l1_idx], target.iloc[val_l1_idx]
        # Compute Optimal l1 round
        opt_l1_rnd, opt_score = compute_optimal_round(trn_l1_x, trn_l1_y, val_l1_x, val_l1_y)
        # print("opt_l1_rnd : ", opt_l1_rnd)
        # Split level2 data
        opt_l2_rnd = []
        fold_lev2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for trn_l2_idx, val_l2_idx in fold_lev2.split(trn_l1_y, trn_l1_y):
            trn_l2_x, trn_l2_y = trn_l1_x.iloc[trn_l2_idx], trn_l1_y.iloc[trn_l2_idx]
            val_l2_x, val_l2_y = trn_l1_x.iloc[val_l2_idx], trn_l1_y.iloc[val_l2_idx]
            # Compute optimal round for current fold
            opt_fold_round, _ = compute_optimal_round(trn_l2_x, trn_l2_y, val_l2_x, val_l2_y)
            opt_l2_rnd.append(opt_fold_round)
        # Print rounds
        values[i, :] = [opt_l1_rnd, np.mean(opt_l2_rnd), np.min(opt_l2_rnd), np.max(opt_l2_rnd)]
        elapsed = (time.time() - start) / 60
        score_mean, score_min, score_max = compute_score(
            trn_l1_x, trn_l1_y, val_l1_x, val_l1_y, 
            rounds = [np.mean(opt_l2_rnd),  
                      np.min(opt_l2_rnd), 
                      np.max(opt_l2_rnd)])
        scores[i, :] = [opt_score, score_mean, score_min, score_max]
        
        print("Opt round %5d / Mean round %5d / Min round %5d / Max round %5d [in %5.1f min]"
              % (values[i, 0], values[i, 1], values[i, 2], values[i, 3], elapsed))
        print("Opt score %.3f / Mean score %.3f / Min score %.3f / Max score %.3f [in %5.1f min]"
              % (scores[i, 0], scores[i, 1], scores[i, 2], scores[i, 3], elapsed))
        
        i += 1


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.figure(figsize=(10,10))
sns.distplot(100 * (values[:, 1] - values[:, 0]) / values[:, 0], label="Mean CV rounds - Optimum round")
sns.distplot(100 * (values[:, 2] - values[:, 0]) / values[:, 0], label="Min CV rounds - Optimum round")
sns.distplot(100 * (values[:, 3] - values[:, 0]) / values[:, 0], label="Max CV rounds - Optimum round")
plt.legend(loc="upper right")
plt.title("Error in Optimum round estimation using 5-CV best rounds (in %)")


# On average the mean of fold rounds gives a good estimate but variance is a big issue here.
# 
# But What about scores ?

# In[ ]:


plt.figure(figsize=(10,10))
sns.distplot(100 * (scores[:, 1] - scores[:, 0]) / scores[:, 0], label="Mean CV rounds - Optimum round")
sns.distplot(100 * (scores[:, 2] - scores[:, 0]) / scores[:, 0], label="Min CV rounds - Optimum round")
sns.distplot(100 * (scores[:, 3] - scores[:, 0]) / scores[:, 0], label="Max CV rounds - Optimum round")
plt.legend(loc="upper right")
plt.title("Error in Optimum score estimation using 5-CV best rounds (in %)")


# In terms of score, mean of round has a far better shape than min or max.
