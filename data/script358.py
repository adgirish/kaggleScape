
# coding: utf-8

# Here is an example of XGBoost hyperparameter tuning by doing a grid search. For reasons of expediency, the notebook will run only a **[randomized grid search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)**. However, I will provide a code for brute-force **[grid search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)** as well - you only need to uncomment that portion if you decide to run this notebook locally.
# 
# Although random grid search will work most of the time if you test at least 20-50 different parameter combinations, I **STRONGLY** recommend that you use something like **[Bayesian optimization](https://github.com/fmfn/BayesianOptimization)** for hyperparameter searching as it is more comprehensive and time-efficient. I will try to publish an example of it later. 
# 

# In[ ]:


__author__ = 'Tilii: https://kaggle.com/tilii7' 

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


# This is a simple timer function that I use in most scripts. I like to know how long things take :-)

# In[ ]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


train_df = pd.read_csv('../input/train.csv', dtype={'id': np.int32, 'target': np.int8})
Y = train_df['target'].values
X = train_df.drop(['target', 'id'], axis=1)
test_df = pd.read_csv('../input/test.csv', dtype={'id': np.int32})
test = test_df.drop(['id'], axis=1)


# Let's set up a parameter grid that will be explored during the search. Note that you can use fewer parameters and fewer options for each parameter. Same goes for more parameter and more options if you want to be very thorough. Also, you can plug in any other ML method instead of XGBoost and search for its optimal parameters.

# In[ ]:


# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }


# A total number of combinations for the set of parameters above is a product of options for each parameter (3 x 5 x 3 x 3 x 3 = 405). It also needs to be multiplied by 5 to calculate a total number of data-fitting runs as we will be doing 5-fold cross-validation. That gets to be a large number in a hurry if you are using many parameters and lots of options, which is why **brute-force grid search takes a long time**.
# 
# Next we set up our classifier. We use sklearn's API of XGBoost as that is a requirement for grid search (another reason why Bayesian optimization may be preferable, as it does not need to be sklearn-wrapped). You should consider setting a learning rate to smaller value (at least 0.01, if not even lower), or make it a hyperparameter for grid searching. I am not using very small value here to save on running time. 
# 
# *Even though we have 4 threads available per job on Kaggle, I think it is more efficient to do XGBoost runs on single threads, but instead run 4 parallel jobs in the grid search. It's up to you whether you want to change this.*

# In[ ]:


xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                    silent=True, nthread=1)


# Next we set up our stratified folds and grid search parameters. I am using AUC as a scoring function, but you can plug in a custom scoring function here if you wish. Grid search wil spawn 4 jobs running a single thread each. The param_comb parameter declares how many different combinations should be picked randomly out of our total (405, see above). I am doing only 5 here, knowing that it will not properly sample the parameter space. Definitely use a bigger number for param_comb.
# 
# *You may want to increase/decrease verbosity depending on your preference.*
# 
# **Note that I have set the number of splits/folds to 3 in order to save time. You should probably put 5 there to get a more reliable result.**

# In[ ]:


folds = 3
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )

# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(X, Y)
timer(start_time) # timing ends here for "start_time" variable


# You can actually follow along as the search goes on. To convert to normalized gini, multiply the obtained AUC values by 2 and subtract 1.
# 
# Let's print the grid-search results and save them in a file.

# In[ ]:


print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=False)


# Not surprisingly, this search does not produce a great score because of 3-fold validation and limited parameter sampling.
# 
# Lastly, let's make a prediction based on best parameters found during the search.

# In[ ]:


y_test = random_search.predict_proba(test)
results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})
results_df.to_csv('submission-random-grid-search-xgb-porto-01.csv', index=False)


# Below is the code for brute-force grid search. It is commented out as it would never finish on Kaggle in 1 hour.

# In[ ]:


# grid = GridSearchCV(estimator=xgb, param_grid=params, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3 )
# grid.fit(X, Y)
# print('\n All results:')
# print(grid.cv_results_)
# print('\n Best estimator:')
# print(grid.best_estimator_)
# print('\n Best score:')
# print(grid.best_score_ * 2 - 1)
# print('\n Best parameters:')
# print(grid.best_params_)
# results = pd.DataFrame(grid.cv_results_)
# results.to_csv('xgb-grid-search-results-01.csv', index=False)

# y_test = grid.best_estimator_.predict_proba(test)
# results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})
# results_df.to_csv('submission-grid-search-xgb-porto-01.csv', index=False)


# Let me know your thoughts or if you have any questions.
# 
