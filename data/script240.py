
# coding: utf-8

# This notebook shows how to use LightGBM in a Regularized RandomForest fashion
# Advantages of LightGBM are :
# - a lot faster than the well-known scikit learn RandomForestClassifier
# - it allows L1 and L2 regularization to achieve better performance
# 
# This notebook is not about finding the best parameters but focuses on using powerfull algos to find important features. 

# In[ ]:


from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# do not display LGBM categorical override warning 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ### Define Gini metrics 
# Let's use the [Extremely Fast Gini Computation by @CPMP](http://www.kaggle.com/cpmpml/extremely-fast-gini-computation)

# In[ ]:


from numba import jit

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


# ### Read data and target

# In[ ]:


trn_df = pd.read_csv("../input/train.csv", index_col=0)

target = trn_df.target
del trn_df["target"]


# ### Define LGBM Random Forest L1 regularizer

# In[ ]:


reg = lgb.LGBMClassifier(boosting_type="rf",
                         num_leaves=165,
                         colsample_bytree=.5,
                         n_estimators=400,
                         min_child_weight=5,
                         min_child_samples=10,
                         subsample=.632, # Standard RF bagging fraction
                         subsample_freq=1,
                         min_split_gain=0,
                         reg_alpha=10, # Hard L1 regularization
                         reg_lambda=0,
                         n_jobs=3)


# ### Run a 5 fold CV and display selected features

# In[ ]:


# do not display LGBM categorical override warning 
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

imp_l1 = pd.DataFrame()
imp_l1["feature"] = trn_df.columns

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
oof_1 = np.zeros(len(trn_df))
start = time.time()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(trn_df.values, target.values)):
    reg.fit(
        trn_df.iloc[trn_idx].values, target.iloc[trn_idx].values,
        feature_name=list(trn_df.columns),
        categorical_feature=[f for f in trn_df.columns if '_cat' in f],
    )
    trn_gini = eval_gini(target.iloc[trn_idx].values,
                               reg.predict_proba(trn_df.iloc[trn_idx].values)[:, 1])
    oof_1[val_idx] = reg.predict_proba(trn_df.iloc[val_idx])[:, 1]
    val_gini = eval_gini(target.iloc[val_idx], oof_1[val_idx])
    imp_l1["imp" + str(fold_ + 1)] = reg.feature_importances_
    print("Gini score for fold %2d : TRN %.6f / VAL %.6f in [%5.1f]" 
          % (fold_, trn_gini, val_gini, (time.time() - start) / 60))

print("OOF score : %.6f in [%5.1f]"
      % (eval_gini(target, oof_1), (time.time() - start) / 60))

# Compute average importances
imps = [f for f in imp_l1 if "imp" in f]
imp_l1["score"] = imp_l1[imps].mean(axis=1)
imp_l1["score"] = 100 * imp_l1["score"] / imp_l1["score"].max()
imp_l1.sort_values("score", ascending=False, inplace=True)

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(10, 8))

# Plot the total crashes
sns.set_color_codes("pastel")

sns.barplot(x="score", y="feature", 
            data=imp_l1[imp_l1["score"] > 0],
            palette=mpl.cm.ScalarMappable(cmap='viridis_r').to_rgba((imp_l1["score"])))
plt.xlabel("LGBM Importance score over 5 folds")
plt.ylabel("Feature")
plt.title("LGBM L1 Regularized Random Forest importances")


# ### Define LGBM Random Forest L2 regularizer

# In[ ]:


reg = lgb.LGBMClassifier(boosting_type="rf",
                         num_leaves=165,
                         colsample_bytree=.5,
                         n_estimators=400,
                         min_child_weight=5,
                         min_child_samples=10,
                         subsample=.632,
                         subsample_freq=1,
                         min_split_gain=0,
                         reg_alpha=0,
                         reg_lambda=5, # L2 regularization
                         n_jobs=3)


# ### Run a 5 fold CV and display L2 selected features

# In[ ]:


imp_l2 = pd.DataFrame()
imp_l2["feature"] = trn_df.columns

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=15)
oof_1 = np.zeros(len(trn_df))
start = time.time()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(trn_df.values, target.values)):
    reg.fit(
        trn_df.iloc[trn_idx].values, target.iloc[trn_idx].values,
        feature_name=list(trn_df.columns),
        categorical_feature=[f for f in trn_df.columns if '_cat' in f],
    )
    trn_gini = eval_gini(target.iloc[trn_idx].values,
                               reg.predict_proba(trn_df.iloc[trn_idx].values)[:, 1])
    oof_1[val_idx] = reg.predict_proba(trn_df.iloc[val_idx])[:, 1]
    val_gini = eval_gini(target.iloc[val_idx], oof_1[val_idx])
    imp_l2["imp" + str(fold_ + 1)] = reg.feature_importances_
    print("Gini score for fold %2d : TRN %.6f / VAL %.6f in [%5.1f]" 
          % (fold_, trn_gini, val_gini, (time.time() - start) / 60))

print("OOF score : %.6f in [%5.1f]"
      % (eval_gini(target, oof_1), (time.time() - start) / 60))

imps = [f for f in imp_l2.columns if "imp" in f]
imp_l2["score"] = imp_l2[imps].mean(axis=1)
imp_l2["score"] = 100 * imp_l2["score"] / imp_l2["score"].max()
imp_l2.sort_values("score", ascending=False, inplace=True)

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(10, 8))

# Plot the total crashes
sns.set_color_codes("pastel")

sns.barplot(x="score", y="feature", 
            data=imp_l2[imp_l2["score"] > 0],
            palette=mpl.cm.ScalarMappable(cmap='viridis_r').to_rgba((imp_l2["score"])))
plt.xlabel("LGBM Importance score over 5 folds")
plt.ylabel("Feature")
plt.title("LGBM L2 Regularized Random Forest importances")

