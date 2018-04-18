
# coding: utf-8

# In[ ]:


MAX_ROUNDS = 650
OPTIMIZE_ROUNDS = False
LEARNING_RATE = 0.05


# I recommend initially setting <code>MAX_ROUNDS</code> fairly high and using <code>OPTIMIZE_ROUNDS</code> to get an idea of the appropriate number of rounds (which, in my judgment, should be close to the maximum value of the optimized <code>tree_count</code> among all folds, maybe even a bit higher if your model is adequately regularized...or maybe not:  it's also a good idea to uncomment <code>verbose=True</code> sometimes and look at the pattern of validation scores for each fold to see what values might work well for all folds).  For a submission run I would turn off <code>OPTIMIZE_ROUNDS</code> and set <code>MAX_ROUNDS</code> to the appropraite number of total rounds.  The problem with "early stopping" by choosing the best round for each fold is that it overfits to the validation data.   It's therefore liable not to produce the optimal model for predicting test data, and if it's used to produce validation data for stacking/ensembling with other models, it would cause this one to have too much weight in the ensemble.
# 
# Also, CatBoost is notoriously slow.  If you want to run it in Kaggle's environment, you need to set a fairly high learning rate so that it will get to a decent fit reasonably quickly.  If you want to use it to do well in this competition, you need to set a lower learning rate and run it for a long time (higher <code>MAX_ROUNDS</code>).  That means you either need to run it overnight or run it on a fast computer with multiple cores (or both).

# In[ ]:


import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from numba import jit


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


# Read data
train_df = pd.read_csv('../input/train.csv', na_values="-1") # .iloc[0:200,:]
test_df = pd.read_csv('../input/test.csv', na_values="-1")


# In[ ]:


# Process data
id_test = test_df['id'].values
id_train = train_df['id'].values

train_df = train_df.fillna(999)
test_df = test_df.fillna(999)

col_to_drop = train_df.columns[train_df.columns.str.startswith('ps_calc_')]
train_df = train_df.drop(col_to_drop, axis=1)  
test_df = test_df.drop(col_to_drop, axis=1)  

for c in train_df.select_dtypes(include=['float64']).columns:
    train_df[c]=train_df[c].astype(np.float32)
    test_df[c]=test_df[c].astype(np.float32)
for c in train_df.select_dtypes(include=['int64']).columns[2:]:
    train_df[c]=train_df[c].astype(np.int8)
    test_df[c]=test_df[c].astype(np.int8)
    
y = train_df['target']
X = train_df.drop(['target', 'id'], axis=1)
y_valid_pred = 0*y
X_test = test_df.drop(['id'], axis=1)
y_test_pred = 0


# In[ ]:


# Set up folds
K = 5
kf = KFold(n_splits = K, random_state = 1, shuffle = True)


# In[ ]:


# Set up classifier
model = CatBoostClassifier(
    learning_rate=LEARNING_RATE, 
    depth=6, 
    l2_leaf_reg = 14, 
    iterations = MAX_ROUNDS,
#    verbose = True,
    loss_function='Logloss'
)


# In[ ]:


# Run CV

for i, (train_index, test_index) in enumerate(kf.split(train_df)):
    
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
    print( "\nFold ", i)
    
    # Run model for this fold
    if OPTIMIZE_ROUNDS:
        fit_model = model.fit( X_train, y_train, 
                               eval_set=[X_valid, y_valid],
                               use_best_model=True
                             )
        print( "  N trees = ", model.tree_count_ )
    else:
        fit_model = model.fit( X_train, y_train )
        
    # Generate validation predictions for this fold
    pred = fit_model.predict_proba(X_valid)[:,1]
    print( "  Gini = ", eval_gini(y_valid, pred) )
    y_valid_pred.iloc[test_index] = pred
    
    # Accumulate test set predictions
    y_test_pred += fit_model.predict_proba(X_test)[:,1]
    
y_test_pred /= K  # Average test set predictions

print( "\nGini for full training set:" )
eval_gini(y, y_valid_pred)


# In[ ]:


# Save validation predictions for stacking/ensembling
val = pd.DataFrame()
val['id'] = id_train
val['target'] = y_valid_pred.values
val.to_csv('cat_valid.csv', float_format='%.6f', index=False)


# In[ ]:


# Create submission file
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_test_pred
sub.to_csv('cat_submit.csv', float_format='%.6f', index=False)


# CV scores certainly seem to be correlated with public LB scores, but the correlation is not nearly as strong as one might hope. The difference so far ranges from .0019 to .0032 for this script without <code>OPTIMIZE_ROUNDS</code> and can be higher when <code>OPTIMIZE_ROUNDS</code> is set (which is to be expected, since that parameter causes the fit to select on each fold for the best CV performance without generally achieving a comparable improvement in LB performance).  Versions 17 through 20 (substantively identical) have slightly better public LB performance than earlier versions, when I sort my submissions by score.  (If one wants to be optimistic about the less significant digits of the LB score, maybe a difference of .0019 is really .0024, and a difference of .0032 is really .0027, so maybe the CV is working better than it first appears.)
