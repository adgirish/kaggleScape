
# coding: utf-8

# # Bayesian hyperparameter tuning of xgBoost
# I took the features generated in my [previous notebook](https://www.kaggle.com/nanomathias/feature-engineering-importance-testing) and tried to tune a xgBoost model to those features using bayesian optimization. I ran the code locally for a few days, after which it had found some good parameters which gave a score of 0.9769 (using all the features of the previous notebook) - this worked out well for me, since I do not have a lot of time to actively work on trying out different models etc, but letting a script like this run for a few days to find good parameters is easy :)

# ## Example 1: xgBoost Parameter Tuning with Scikit-Optimize
# The following code is exactly what I used to tune the parameters - only difference is that I ran with the last 20 million samples in training set, and used the features from [previous notebook](https://www.kaggle.com/nanomathias/feature-engineering-importance-testing) instead of the raw features as done here. First I'll load the needed libraries and data.

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

# SETTINGS - CHANGE THESE TO GET SOMETHING MEANINGFUL
ITERATIONS = 10 # 1000
TRAINING_SIZE = 100000 # 20000000
TEST_SIZE = 25000

# Load data
X = pd.read_csv(
    '../input/train.csv', 
    skiprows=range(1,184903891-TRAINING_SIZE), 
    nrows=TRAINING_SIZE,
    parse_dates=['click_time']
)

# Split into X and y
y = X['is_attributed']
X = X.drop(['click_time','is_attributed', 'attributed_time'], axis=1)


# To do the bayesian parameter tuning, I use the [BayesSearchCV](https://scikit-optimize.github.io/#skopt.BayesSearchCV) class of scikit-optimize. It works basically as a drop-in replacement for GridSearchCV and RandomSearchCV, but generally I get better results with it. In the following I define the BayesSearchCV object, and write a short convenience function that will be used during optimization to output current status of the tuning. Locally I have access to more cores and run with n_jobs=4 for the classifier, and n_jobs=6 for the BayesSearchCV object.

# In[2]:


# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = xgb.XGBClassifier(
        n_jobs = 1,
        objective = 'binary:logistic',
        eval_metric = 'auc',
        silent=1,
        tree_method='approx'
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")


# Finally, let the parameter tuning run and wait for good results :)

# In[3]:


# Fit the model
result = bayes_cv_tuner.fit(X.values, y.values, callback=status_print)


# ## Example 2: lightGBM Parameter Tuning with Scikit-Optimize
# I have not myself submitted any models run with lightGBM as of yet, but here is an example of how to run the parameter search with lightGBM instead of xgBoost

# In[4]:


# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = lgb.LGBMRegressor(
        objective='binary',
        metric='auc',
        n_jobs=1,
        verbose=0
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (1, 100),      
        'max_depth': (0, 50),
        'min_child_samples': (0, 50),
        'max_bin': (100, 1000),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        'subsample_for_bin': (100000, 500000),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (50, 100),
    },    
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)

# Fit the model
result = bayes_cv_tuner.fit(X.values, y.values, callback=status_print)


# ## Example 3: Different cross-validators
# Some people have asked about CV strategy, and as seen I've just used the basic Stratified K-fold strategy; that was however mostly due to time constraints, and thus me not thinking that much about it. There are a lot of potentially better options, especially considering the temporal nature of this problem. Adding these are really easy using scikit-learn cross-validators; you just plug-n-play a new cross-validator into the `cv = ` options of BayesSearchCV. Examples could be a single train-test split, where we e.g. use one day for training, and one for testing (adjust accordingly):

# In[6]:


from sklearn.model_selection import PredefinedSplit

# Training [index == -1], testing [index == 0])
test_fold = np.zeros(len(X))
test_fold[:(TRAINING_SIZE-TEST_SIZE)] = -1
cv = PredefinedSplit(test_fold)

# Check that we only have a single train-test split, and the size
train_idx, test_idx = next(cv.split())
print(f"Splits: {cv.get_n_splits()}, Train size: {len(train_idx)}, Test size: {len(test_idx)}")


# Alternatively, we could want to use the [TimeSeriesSplit](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit) cross-validator, which allows us to do several "into the future folds" for predictions

# In[14]:


from sklearn.model_selection import TimeSeriesSplit

# Here we just do 3-fold timeseries CV
cv = TimeSeriesSplit(max_train_size=None, n_splits=3)

# Let us check the sizes of the folds. Note that you can keep train size constant with max_train_size if needed
for i, (train_index, test_index) in enumerate(cv.split(X)):
    print(f"Split {i+1} / {cv.get_n_splits()}:, Train size: {len(train_index)}, Test size: {len(test_index)}")


# ## Optimal xgBoost parameters
# ![](http://)After a few days of running for xgBoost, it found the following optimal parameters. Again, note that these gave me a 0.9769 score on [these features](https://www.kaggle.com/nanomathias/feature-engineering-importance-testing) and not the raw features, by training on the entire training set.

# In[ ]:


{
    'colsample_bylevel': 0.1,
    'colsample_bytree': 1.0,
    'gamma': 5.103973694670875e-08,
    'learning_rate': 0.140626707498132,
    'max_delta_step': 20,
    'max_depth': 6,
    'min_child_weight': 4,
    'n_estimators': 100,
    'reg_alpha': 1e-09,
    'reg_lambda': 1000.0,
    'scale_pos_weight': 499.99999999999994,
    'subsample': 1.0
}

