
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


# Preprocessing (remove ps_calc cols, replace -1 with NaN, OHE categorical features)
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

for column in cat_features:
    temp = pd.get_dummies(pd.Series(train[column]))
    train = pd.concat([train,temp],axis=1)
    train = train.drop([column],axis=1)
    
for column in cat_features:
    temp = pd.get_dummies(pd.Series(test[column]))
    test = pd.concat([test,temp],axis=1)
    test = test.drop([column],axis=1)

print(train.values.shape, test.values.shape)


# In[ ]:


class Ensemble(object):    
    def __init__(self, mode, n_splits, stacker_2, stacker_1, base_models):
        self.mode = mode
        self.n_splits = n_splits
        self.stacker_2 = stacker_2
        self.stacker_1 = stacker_1
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)


        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                                                             random_state=2016).split(X, y))
        
        OOF_columns = []

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):                
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]

                print ("Fit %s_%d fold %d" % (str(clf).split("(")[0], i+1, j+1))
                clf.fit(X_train, y_train)

                S_train[test_idx, i] = clf.predict_proba(X_holdout)[:,1]  
                S_test_i[:, j] = clf.predict_proba(T)[:,1]                
            S_test[:, i] = S_test_i.mean(axis=1)
            
            print("  Base model_%d score: %.5f\n" % (i+1, roc_auc_score(y, S_train[:,i])))
        
            OOF_columns.append('Base model_'+str(i+1))
        OOF_S_train = pd.DataFrame(S_train, columns = OOF_columns)
        print('\n')
        print('Correlation between out-of-fold predictions from Base models:')
        print('\n')
        print(OOF_S_train.corr())
        print('\n')
            
        
        if self.mode==1:
            
            folds_2 = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                                                                   random_state=2016).split(S_train, y))
            
            OOF_columns = []

            S_train_2 = np.zeros((S_train.shape[0], len(self.stacker_1)))
            S_test_2 = np.zeros((S_test.shape[0], len(self.stacker_1)))
            
            for i, clf in enumerate(self.stacker_1):
            
                S_test_i_2 = np.zeros((S_test.shape[0], self.n_splits))

                for j, (train_idx, test_idx) in enumerate(folds_2):
                    X_train_2 = S_train[train_idx]
                    y_train_2 = y[train_idx]
                    X_holdout_2 = S_train[test_idx]

                    print ("Fit %s_%d fold %d" % (str(clf).split("(")[0], i+1, j+1))
                    clf.fit(X_train_2, y_train_2)
                                 
                    S_train_2[test_idx, i] = clf.predict_proba(X_holdout_2)[:,1] 
                    S_test_i_2[:, j] = clf.predict_proba(S_test)[:,1]
                S_test_2[:, i] = S_test_i_2.mean(axis=1)
                
                print("  1st level model_%d score: %.5f\n"%(i+1,
                                                            roc_auc_score(y, S_train_2.mean(axis=1))))
                
                OOF_columns.append('1st level model_'+str(i+1))
            OOF_S_train = pd.DataFrame(S_train_2, columns = OOF_columns)
            print('\n')
            print('Correlation between out-of-fold predictions from 1st level models:')
            print('\n')
            print(OOF_S_train.corr())
            print('\n')


        if self.mode==2:
            
            WOC_columns = []
        
            S_train_2 = np.zeros((S_train.shape[0], len(self.stacker_1)))
            S_test_2 = np.zeros((S_test.shape[0], len(self.stacker_1)))
               
            for i, clf in enumerate(self.stacker_1):
            
                S_train_i_2= np.zeros((S_train.shape[0], S_train.shape[1]))
                S_test_i_2 = np.zeros((S_test.shape[0], S_train.shape[1]))
                                       
                for j in range(S_train.shape[1]):
                                
                    S_tr = S_train[:,np.arange(S_train.shape[1])!=j]
                    S_te = S_test[:,np.arange(S_test.shape[1])!=j]
                                               
                    print ("Fit %s_%d subset %d" % (str(clf).split("(")[0], i+1, j+1))
                    clf.fit(S_tr, y)

                    S_train_i_2[:, j] = clf.predict_proba(S_tr)[:,1]                
                    S_test_i_2[:, j] = clf.predict_proba(S_te)[:,1]
                S_train_2[:, i] = S_train_i_2.mean(axis=1)    
                S_test_2[:, i] = S_test_i_2.mean(axis=1)
            
                print("  1st level model_%d score: %.5f\n"%(i+1,
                                                            roc_auc_score(y, S_train_2.mean(axis=1))))
                
                WOC_columns.append('1st level model_'+str(i+1))
            WOC_S_train = pd.DataFrame(S_train_2, columns = WOC_columns)
            print('\n')
            print('Correlation between without-one-column predictions from 1st level models:')
            print('\n')
            print(WOC_S_train.corr())
            print('\n')
            
            
        try:
            num_models = len(self.stacker_2)
            if self.stacker_2==(et_model):
                num_models=1
        except TypeError:
            num_models = len([self.stacker_2])
            
        if num_models==1:
                
            print ("Fit %s for final\n" % (str(self.stacker_2).split("(")[0]))
            self.stacker_2.fit(S_train_2, y)
            
            stack_res = self.stacker_2.predict_proba(S_test_2)[:,1]
        
            stack_score = self.stacker_2.predict_proba(S_train_2)[:,1]
            print("2nd level model final score: %.5f" % (roc_auc_score(y, stack_score)))
                
        else:
            
            F_columns = []
            
            stack_score = np.zeros((S_train_2.shape[0], len(self.stacker_2)))
            res = np.zeros((S_test_2.shape[0], len(self.stacker_2)))
            
            for i, clf in enumerate(self.stacker_2):
                
                print ("Fit %s_%d" % (str(clf).split("(")[0], i+1))
                clf.fit(S_train_2, y)
                
                stack_score[:, i] = clf.predict_proba(S_train_2)[:,1]
                print("  2nd level model_%d score: %.5f\n"%(i+1,roc_auc_score(y, stack_score[:, i])))
                
                res[:, i] = clf.predict_proba(S_test_2)[:,1]
                
                F_columns.append('2nd level model_'+str(i+1))
            F_S_train = pd.DataFrame(stack_score, columns = F_columns)
            print('\n')
            print('Correlation between final predictions from 2nd level models:')
            print('\n')
            print(F_S_train.corr())
            print('\n')
        
            stack_res = res.mean(axis=1)            
            print("2nd level models final score: %.5f" % (roc_auc_score(y, stack_score.mean(axis=1))))

            
        return stack_res


# In[ ]:


# LightGBM params
lgb_params_1 = {
    'learning_rate': 0.02,
    'n_estimators': 750,
    'subsample': 0.8,
    'subsample_freq': 10,
    'colsample_bytree': 0.8,
    'max_bin': 10,
    'min_child_samples': 500,
    'seed': 99
}

lgb_params_2 = {
    'learning_rate': 0.02,
    'n_estimators': 1200,
    'subsample': 0.7,
    'subsample_freq': 2,
    'colsample_bytree': 0.3,  
    'num_leaves': 16,
    'seed': 99
}

lgb_params_3 = {
    'learning_rate': 0.02,
    'n_estimators': 475,
    'subsample': 0.4,
    'subsample_freq': 1,
    'colsample_bytree': 0.9,  
    'num_leaves': 28,
    'max_bin': 10,
    'min_child_samples': 700,
    'seed': 99
}


# In[ ]:


# Base models
lgb_model_1 = LGBMClassifier(**lgb_params_1)

lgb_model_2 = LGBMClassifier(**lgb_params_2)

lgb_model_3 = LGBMClassifier(**lgb_params_3)


# In[ ]:


# Stacker models
log_model = LogisticRegression()

et_model = ExtraTreesClassifier(n_estimators=100, max_depth=6, min_samples_split=10, random_state=10)

mlp_model = MLPClassifier(max_iter=7, random_state=42)


# In[ ]:


# Mode 2 run
stack = Ensemble(mode=2,
        n_splits=3,
        stacker_2 = (log_model, et_model),         
        stacker_1 = (log_model, et_model, mlp_model),
        base_models = (lgb_model_1, lgb_model_2, lgb_model_3))       
        
y_pred = stack.fit_predict(train, target_train, test)  


# In[ ]:


# Submission from mode 2
sub = pd.DataFrame()
sub['id'] = id_test
sub['target'] = y_pred
sub.to_csv('2-level_stacked_1.csv', index=False)

