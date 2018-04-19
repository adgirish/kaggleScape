
# coding: utf-8

# ## Parameter tuning and feature engineering for highly imbalanced data
# 
# Introduction -
# This includes a brief EDA with details about dealing with highly imbalanced data. This notebook is a living document -- that is, I will be updating it on a daily basis with additional experiments. 
# 
# Things that I tried differently (so-far) :
# 1. Down-sampling of unbalanced data (True/False) to 1:10, 1:15, 1:20 ratio. Original is ~ 1:25. 
# 2. Tried a few iterations for RandomSearchCV (Random Forest sofar) with different sub-samples (10%) for parameter tuning.  
# 3. Addition of new feature from ps_car_13 (ps_car_13^2*90,000) 

# ### Importing libararies

# In[ ]:


## Importing packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split

import random
import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.integrate


# ### Functions that I will be using later in the code (gini code taken from some other Kernel) 

# #### Defining gini for prediction score

# In[ ]:



def gini(truth, predictions):
    assert (len(truth) == len(predictions))
    all = np.asarray(np.c_[truth, predictions, np.arange(len(truth))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(truth) + 1) / 2.
    return giniSum / len(truth)

def gini_xgb(predictions, truth):
    truth = truth.get_label()
    return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)

def gini_lgb(truth, predictions):
    score = gini(truth, predictions) / gini(truth, truth)
    return 'gini', score, True

def gini_sklearn(truth, predictions):
    return abs(gini(truth, predictions) / gini(truth, truth))

gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)



# ### Preprocessing of data

# In[ ]:


traincsv = pd.read_csv("../input/train.csv") # reading csv 


# In[ ]:


train = pd.DataFrame(traincsv) # pandas df


# In[ ]:


train.shape # exploring shape


# In[ ]:


train.columns # exploring the columns 


# In[ ]:


train.iloc[:5,]


# In[ ]:


train.describe(include = 'all')


# In[ ]:


## distribution of target variable (insurance claim filed or not)

import matplotlib.pyplot as plt
plt.hist(train['target'])
plt.show()

print('Percentage of claims filed :' , str(np.sum(train['target'])/train.shape[0]*100), '%')


# In[ ]:


## Distribution of columns based on groups

cols = train.columns

count_ind = 0; count_cat = 0; count_calc = 0; count_car = 0; count_remain = 0

for col in cols:
    col = str(col)
    if 'ind' in col:
        count_ind += 1
    elif 'cat' in col:
        count_cat += 1
    elif 'calc' in col:
        count_calc += 1
    elif 'car' in col:
        count_car += 1
    else:
        count_remain += 1

print("Total columns are", train.shape[1])
print("Total ind columns: ", count_ind, "| Percentage:", round(count_ind*100/train.shape[1],1))
print("Total cat columns: ", count_cat, "| Percentage:", round(count_cat*100/train.shape[1],1))
print("Total calc columns: ", count_calc, "| Percentage:", round(count_calc*100/train.shape[1],1))
print("Total car columns: ", count_car, "| Percentage:", round(count_car*100/train.shape[1],1))
print("Total other columns: ", count_remain, "| Percentage:", round(count_remain*100/train.shape[1],1))


# In[ ]:


## Perentage of missing values in each column. (Only columns with % > 0)

missing = np.sum(train == -1)/train.shape[0]*100
print("Percentage of missing values (denoted by -1)")
print(missing[missing > 0].sort_values(ascending = False))


# I will not change anything in **categorical** columns with null. But will add new columns of **isNull** (0,1) in those. For continous variables, -1 is better to be replaced by **median**. Let's do that below.

# In[ ]:


# making a copy
train_natreat = train.copy()
train_natreat.shape


# In[ ]:


#train_natreat.columns


# ### Transformations
# 
# * Adding dummy columns "IsNA?" with 0/1 in rows for each variable with NAs  
# * Replacing -1 with median for *continuous* variables
# * Adding a column with sum of NAs for each row
# * By running random forest, **ps_car_13** was found to be one of the most important feature. Adding a transformation = round(ps_car_13^2*90000 , 2). [Thanks to "raddar" - https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41489]

# In[ ]:


## treating missing values. (denoted by -1 in data)
from statistics import median

# categorical NA variables
cat_na_cols = ['ps_car_07_cat', 'ps_car_09_cat','ps_car_01_cat', 'ps_car_02_cat' ,
              'ps_ind_05_cat', 'ps_ind_02_cat', 'ps_ind_04_cat']

# adding dummy isNA variable for each feature with atleast 1 NA (-1)
for l in cat_na_cols:
    #train[l] = train[l].replace(-1, np.nan)
    new_col_name = str(l) + "isNull"
    train_natreat[new_col_name] = (train_natreat[l] == -1).astype('int')
    print('done')
    
num_na_cols = ['ps_reg_03', 'ps_car_14', 'ps_car_11', 'ps_car_12' ]
for l in num_na_cols:
    new_col_name = str(l) + "isNull"
    train_natreat[new_col_name] = (train_natreat[l] == -1).astype('int')
    train_natreat[l] = train_natreat[l].replace(-1, train_natreat[l].median())

# column with sum of NA in each row
train_natreat["sumNA"] = (train_natreat == -1).astype(int).sum(axis=1)

## Another important feature with transformation of ps_car_13
train_natreat['ps_car_13_new1'] = round(train_natreat['ps_car_13']*train_natreat['ps_car_13']*90000,2)
train_natreat.columns

print("new columns are - ", train_natreat.columns)


# ##### Skipping logistic regression

# In[ ]:


# # split data into X and Y
# X = train_nacleaned.iloc[:,2:train_nacleaned.shape[1]]
# Y = train_nacleaned.iloc[:,1]

# # scaling model to feed into logistic regression
# X_scaled = scale(X)

# # Split data into train (80%) /test (20%) set 
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.20, random_state = 99)


# In[ ]:


# # Fit logistic regression model
# logit = LogisticRegression(penalty = 'l1', random_state = 99)
# logit.fit(x_train, y_train)


# In[ ]:


# Predict logistic regression on test data and display sensitivity/specificity
# logit_predict = logit.predict(X_test)
# plt.hist(logit_predict)
# plt.show()
# nofaults, val = np.where(y_test==0)
# faults, val = np.where(y_test==1)
# sensitivity = sum(logit_predict[faults])/len(faults)
# logit_predict_nofaults = logit_predict[nofaults]
# specificity = float(len(logit_predict_nofaults[logit_predict_nofaults == 0]))/len(nofaults)
# print "LOGIT: sensitivity =",sensitivity, "| specificity =",specificity


# ## Random Forest Method - 
# 

# In[ ]:


# split data into df and Y
df = train_natreat.drop('target', axis = 1)
y = train_natreat.iloc[:,1]


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df, y, test_size = 0.20, random_state = 99)


# Loading test data 

# In[ ]:


# test = pd.read_csv("test.csv")
# test = pd.DataFrame(test)


# #### Transformations for test set
# 
# Just replicating whatever transformation I did in `train`

# In[ ]:


## treating missing values. (denoted by -1 in data)

# Commenting it out as not uploading test data here. Just 20-80 split

"""
from statistics import median

# categorical NA variables
cat_na_cols = ['ps_car_07_cat', 'ps_car_09_cat','ps_car_01_cat', 'ps_car_02_cat' ,
              'ps_ind_05_cat', 'ps_ind_02_cat', 'ps_ind_04_cat']

## Adding a dummy column IsNA? for all columns with NA. 
# Even if it's not important, it doesn't harm to input these for random forest

for l in cat_na_cols:
    #train[l] = train[l].replace(-1, np.nan)
    new_col_name = str(l) + "isNull"
    test[new_col_name] = (test[l] == -1).astype('int')
    print('done')
    
num_na_cols = ['ps_reg_03', 'ps_car_14', 'ps_car_11', 'ps_car_12' ]
for l in num_na_cols:
    new_col_name = str(l) + "isNull"
    test[new_col_name] = (test[l] == -1).astype('int')
    test[l] = test[l].replace(-1, test[l].median())
    print('done')
    
# column with sum of NA in each row
test["sumNA"] = (test == -1).astype(int).sum(axis=1)

test['ps_car_13_new1'] = round(test['ps_car_13']*test['ps_car_13']*90000,2)
"""


# ### Parameter tuning for Random Forest

# #### Self tuning for parameters without using CV

# Downsampling as data is unbalanced. True ratio is like ~1/25. Let's pick 3 different ratios (but I will take total rows to be = 50,000 to increase CV speed). I hope I don't loose much information. Ratios -
# 1. **1/20** (Trues are 2500/ 50,000)
# 2. **1/15** (Trues are 3333/ 50,000)
# 3. **1/10** (Trues are 5000/ 50,000)

# In[ ]:


# sub sampling for tuning (will work on only 1 lakhs rows of data)
count0 = np.sum(train_natreat['target'] == 0)
count1 = np.sum(train_natreat['target'] == 1)
count0 + count1 == train_natreat.shape[0]


# #### Grid of parameters to tune

# In[ ]:


param_grid = {"n_estimators": np.arange(25, 500, 50,dtype=int),
              "max_depth": np.arange(1, 20, 2),
              #"min_samples_split": np.arange(1,150,1),
              "min_samples_leaf": [1,5,10,50,100,200,500]}
              #"max_leaf_nodes": np.arange(2,60,6),
              #"min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}

# param_grid = {"n_estimators": np.arange(25, 500, 50,dtype=int),
#               "max_depth": np.arange(1, 10, 1)}


# #### Iterating for sub samle of 50000 and 3 different True/False ratios to find best parameters.
# 
# I have commented the below code as it takes 15 minutes to run. But I have hardcoded the parameters/output from cross validation. 

# In[ ]:


features = train_natreat.columns.drop(['id', 'target'], 1)
rf = RandomForestClassifier(random_state=50, n_jobs = -1, oob_score=True)

# for i in range(3):
#     for j in [2500, 3333, 5000]:
#         index0 = train_natreat.index[train_natreat['target'] == 0]
#         index1 = train_natreat.index[train_natreat['target'] == 1]

#         subsample_index0 = random.sample(list(index0),50000 - j)
#         subsample_index1 = random.sample(list(index1),j)
#         subsample_index_both = subsample_index0 + subsample_index1
    
#         train_natreat_sample = train_natreat.ix[subsample_index_both]

#         # splitting y and x
#         #features = train_natreat_sample2.columns.drop(['id', 'target'], 1)
#         df = train_natreat_sample[features]
#         y = train_natreat_sample.iloc[:,1]
              
#         random_cv = RandomizedSearchCV(rf, param_distributions = param_grid, cv= 3, scoring=gini_scorer)
#         random_cv.fit(df, y)
#         print(random_cv.best_score_)
#         print(random_cv.best_params_)
#         print(random_cv.best_estimator_)
    
#score = cross_val_score(rf, df, y, scoring=gini_scorer, cv=StratifiedKFold()).mean()


# ### Hardcoding selected parameters from CV
# 
# -0.21262327096741435
# {'n_estimators': 225, 'min_samples_leaf': 1, 'max_depth': 1}
# 
# -0.2169821613491697
# {'n_estimators': 175, 'min_samples_leaf': 1, 'max_depth': 15}
# 
# -0.21034002745394356
# {'n_estimators': 375, 'min_samples_leaf': 1, 'max_depth': 1}
# 
# -0.18180637478659406
# {'n_estimators': 25, 'min_samples_leaf': 1, 'max_depth': 9}
# 
# -0.18410624559552097
# {'n_estimators': 125, 'min_samples_leaf': 1, 'max_depth': 17}
# 
# -0.21881615663247375
# {'n_estimators': 175, 'min_samples_leaf': 1, 'max_depth': 1}
# 
# -0.20685089187304984
# {'n_estimators': 25, 'min_samples_leaf': 5, 'max_depth': 11}
# 
# -0.19705679411050572
# {'n_estimators': 225, 'min_samples_leaf': 1, 'max_depth': 19}
# 
# -0.22252957405534599
# {'n_estimators': 175, 'min_samples_leaf': 50, 'max_depth': 1}
# 
# #### So, I am selecting {'n_estimators': 175, 'min_samples_leaf': 50, 'max_depth': 10}. 
# But this is only for small sub-sample. Gini should increase for complete sample set.

# Now let's fit the model using tuned parameters and submit predictions 

# In[ ]:


#from sklearn.ensemble import RandomForestClassifier
x_train, x_test, y_train, y_test = train_test_split(train_natreat[features], train_natreat.iloc[:,1], test_size = 0.20, random_state = 99)

rf_final = RandomForestClassifier(criterion='gini',n_jobs=-1,min_samples_leaf=50,
            max_depth=10, n_estimators=175, random_state=50, oob_score = True)
rf_final.fit(x_train, y_train)


# ### Plotting top features

# In[ ]:


objects = x_train[features].columns
performance = rf_final.feature_importances_
features1 = pd.concat([pd.Series(objects),  pd.Series(performance)], axis = 1)
features1.columns = ['feature', 'importance']
features2 = features1.sort_values(by = 'importance', ascending=False)

top15_feautures = features2[0:15]


# In[ ]:


objects = top15_feautures.iloc[:,0]
y_pos = np.arange(len(objects))
performance = top15_feautures.iloc[:,1]
 
plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('Weights')
plt.title('Features')
 
plt.show()


# * **ps_car_13_new1** is the feature that I transfored using ps_car_13. It is coming out to be most important feature now. 
# * None of the feature related to "NA" came out to be important (atleast in top 15)

# #### Prediction on test (remaining 20%) set

# In[ ]:


prediction = rf_final.predict_proba(x_test[features])
plt.hist(prediction[:,1])
plt.show()


# In[ ]:


from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, prediction[:,1])
plt.plot(fpr, tpr,'r')
plt.plot([0,1],[0,1],'b')
plt.title('AUC: {}'.format(auc(fpr,tpr)))
plt.show()


# In[ ]:


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

gini_predictions = gini(y_test, prediction[:,1])
gini_max = gini(y_test, y_test)
ngini= gini_normalized(y_test, prediction[:,1])
print('Gini: %.3f, Max. Gini: %.3f, Normalized Gini: %.3f' % (gini_predictions, gini_max, ngini))

