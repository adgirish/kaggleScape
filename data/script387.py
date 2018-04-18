
# coding: utf-8

# Vladimir Demidov has [an excellent kernel](https://www.kaggle.com/yekenot/simple-stacker-lb-0-284) that shows how to do stacking elegantly and also shows (via its public leaderboard score) why to do stacking.  But there were a couple of things I wanted to do differently, so I decided to release my own version.  Aside from being a notebook, this is almost identical to the original, except for two substantive changes (and a few minor cosmetic ones).
# 
# First, I apply a log-odds transformation to the base models' predictions.  Since the top-level model is a logistic regression, it takes a linear combination of its inputs and then applies a logistic transformation (the inverse of log-odds) to the result.  If the inputs are themselves expressed as probabilities, then it's kind of like doing the logistic transformation twice (and if one of the base models were a logistic regression, it would be exactly like doing the logistic transformation twice), which would be hard to justify.  To put the base model predictions in "units that a linear model understands," I express them as log odds ratios rather than probabilities.
# 
# Second, I fit the logistic regression without an intercept.  Although the intercept might be necessary without the log-odds transformation, the regression with log odds gives reasonable results without it, and it's not clear why it should be there.  In my view, since the gini coefficient depends on order, an added constant has no substantive meaning, and the opportunity to add one is simply an opportunity to overfit.  (If we cared about the actual probabilities, you could make a case for using a constant term as being sort of like adding another base model that always predicts the same number, but here that justification doesn't apply.)
# 
# My model (as you should be able to see by comparing this notebook to his log) produces a very slightly higher CV score ("Stacker score") than Vladimir's.  I haven't submitted the output, but I expect it would get the same public leaderboard score.  But this is one of those cases where you have to make a judgment based on what you think is a better modeling practice rather than what results it gets in limited tests.

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score

# Regularized Greedy Forest
from rgf.sklearn import RGFClassifier     # https://github.com/fukatani/rgf_python


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# Preprocessing 
id_test = test['id'].values
target_train = train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)


col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  


train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)


cat_features = [a for a in train.columns if a.endswith('cat')]

# Make sure both train & test have a full set of dummies
train['intrain'] = True  
test['intrain'] = False
both = pd.concat([train,test],axis=0)
for column in cat_features:
	temp = pd.get_dummies(pd.Series(both[column]))
	both = pd.concat([both,temp],axis=1)
	both = both.drop([column],axis=1)
train = both[both.intrain==True].drop(['intrain'],axis=1)
test = both[both.intrain==False].drop(['intrain'],axis=1)

print(train.values.shape, test.values.shape)


# In[ ]:


class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=self.n_splits, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

            print("     Model score: %.5f\n" % roc_auc_score(y, S_train[:,i]))

        self.base_preds = S_test
        
        # Log odds transformation
        almost_zero = 1e-12
        almost_one = 1 - almost_zero  # To avoid division by zero
        S_train[S_train>almost_one] = almost_one
        S_train[S_train<almost_zero] = almost_zero
        S_train = np.log(S_train/(1-S_train))
        S_test[S_test>almost_one] = almost_one
        S_test[S_test<almost_zero] = almost_zero
        S_test = np.log(S_test/(1-S_test))
        
        results = cross_val_score(self.stacker, S_train, y, cv=self.n_splits, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        print( 'Coefficients:', self.stacker.coef_ )

        res = self.stacker.predict_proba(S_test)[:,1]
        return res


# In[ ]:


# LightGBM params
lgb_params = {}
lgb_params['learning_rate'] = 0.02
lgb_params['n_estimators'] = 650
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8   
lgb_params['min_child_samples'] = 500
lgb_params['random_state'] = 99


lgb_params2 = {}
lgb_params2['n_estimators'] = 1090
lgb_params2['learning_rate'] = 0.02
lgb_params2['colsample_bytree'] = 0.3   
lgb_params2['subsample'] = 0.7
lgb_params2['subsample_freq'] = 2
lgb_params2['num_leaves'] = 16
lgb_params2['random_state'] = 99


lgb_params3 = {}
lgb_params3['n_estimators'] = 1100
lgb_params3['max_depth'] = 4
lgb_params3['learning_rate'] = 0.02
lgb_params3['random_state'] = 99


# RandomForest params
#rf_params = {}
#rf_params['n_estimators'] = 650
#rf_params['max_depth'] = 14
#rf_params['min_samples_split'] = 40
#rf_params['min_samples_leaf'] = 35


# ExtraTrees params
#et_params = {}
#et_params['n_estimators'] = 300
#et_params['max_features'] = .2
#et_params['max_depth'] = 10
#et_params['min_samples_split'] = 7
#et_params['min_samples_leaf'] = 40


# XGBoost params
#xgb_params = {}
#xgb_params['objective'] = 'binary:logistic'
#xgb_params['learning_rate'] = 0.04
#xgb_params['n_estimators'] = 490
#xgb_params['max_depth'] = 4
#xgb_params['subsample'] = 0.9
#xgb_params['colsample_bytree'] = 0.9  
#xgb_params['min_child_weight'] = 10


# CatBoost params
#cat_params = {}
#cat_params['iterations'] = 900
#cat_params['depth'] = 8
#cat_params['rsm'] = 0.95
#cat_params['learning_rate'] = 0.03
#cat_params['l2_leaf_reg'] = 3.5  
#cat_params['border_count'] = 8
#cat_params['gradient_iterations'] = 4


# Regularized Greedy Forest params
#rgf_params = {}
#rgf_params['max_leaf'] = 2000
#rgf_params['learning_rate'] = 0.5
#rgf_params['algorithm'] = "RGF_Sib"
#rgf_params['test_interval'] = 100
#rgf_params['min_samples_leaf'] = 3 
#rgf_params['reg_depth'] = 1.0
#rgf_params['l2'] = 0.5  
#rgf_params['sl2'] = 0.005


# In[ ]:


lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)

#rf_model = RandomForestClassifier(**rf_params)

#et_model = ExtraTreesClassifier(**et_params)
        
#xgb_model = XGBClassifier(**xgb_params)

#cat_model = CatBoostClassifier(**cat_params)

#rgf_model = RGFClassifier(**rgf_params) 

#gb_model = GradientBoostingClassifier(max_depth=5)

#ada_model = AdaBoostClassifier()

log_model = LogisticRegression(fit_intercept=False)


# In[ ]:


stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (lgb_model, lgb_model2, lgb_model3))        
        
y_pred = stack.fit_predict(train, target_train, test)        


# In[ ]:


sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('stacked_1.csv', index=False)

