
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
import lightgbm as lgb
import time

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.max_rows', 1000)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Note to others
# 
# This is the second of a series of notebooks that I am working on for educational purpose, to demonstrate some run-of-the mill techniques we use on Kaggle for a student audience.
# 
# So when most folks here are competiting to overfit the LB, we are doing some small effort to fill our students with handy knowledge and the virtue of solid CV. 
# 
# Anyhow, since our students are also competing as teams in this competition, we decided to use the kernel facility as a way to share knowledge. Nevertheless, all comments are welcome, and let's all enjoy the last days of this competition!

# # Porto Seguo - End-to-end Ensemble
# 
# In this competition we are tasked with making predictive models that can predict if a given driver will make insurance claim. In a ["previous kernel"](https://www.kaggle.com/yifanxie/porto-seguro-tutorial-simple-e2e-pipeline) we have breifly explored the data, did some useful categorical feature encoding, and presented a simple model building pipeline.
# 
# In this kernel, we are going to progress a bit more on the model building aspect. Firstly, we will introduce the technique to generate out-out-fold train and test predictions for several models, we will then use these out-of-fold predictions to be  ensemble model. 
# 
# Strickly speaking, the ensemble method we use here is referred as **Stacked generalization** as very well ilustrated already by the following blog/articles:
# * [*Kaggle Ensemble Guide*](https://mlwave.com/kaggle-ensembling-guide/) by [Triskelion](https://www.kaggle.com/triskelion)
# * [*Stacking Made Easy: An Introduction to StackNet*](http://blog.kaggle.com/2017/06/15/stacking-made-easy-an-introduction-to-stacknet-by-competitions-grandmaster-marios-michailidis-kazanova/) by Competitions Grandmaster [Marios Michailidis (KazAnova)](https://www.kaggle.com/kazanova)
# 
# So without further ado, let's get technical.
# 

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
sample_submission=pd.read_csv('../input/sample_submission.csv')


# 
# # 1. Categorical feature encoding and feature reduction
# The following part is a strict copy & paste from the first kernel, so for detailed explaination please check it out there.
# 
# ## 1.1 Frequency Encoding

# In[ ]:


# This function late in a list of features 'cols' from train and test dataset, 
# and performing frequency encoding. 
def freq_encoding(cols, train_df, test_df):
    # we are going to store our new dataset in these two resulting datasets
    result_train_df=pd.DataFrame()
    result_test_df=pd.DataFrame()
    
    # loop through each feature column to do this
    for col in cols:
        
        # capture the frequency of a feature in the training set in the form of a dataframe
        col_freq=col+'_freq'
        freq=train_df[col].value_counts()
        freq=pd.DataFrame(freq)
        freq.reset_index(inplace=True)
        freq.columns=[[col,col_freq]]

        # merge ths 'freq' datafarme with the train data
        temp_train_df=pd.merge(train_df[[col]], freq, how='left', on=col)
        temp_train_df.drop([col], axis=1, inplace=True)

        # merge this 'freq' dataframe with the test data
        temp_test_df=pd.merge(test_df[[col]], freq, how='left', on=col)
        temp_test_df.drop([col], axis=1, inplace=True)

        # if certain levels in the test dataset is not observed in the train dataset, 
        # we assign frequency of zero to them
        temp_test_df.fillna(0, inplace=True)
        temp_test_df[col_freq]=temp_test_df[col_freq].astype(np.int32)

        if result_train_df.shape[0]==0:
            result_train_df=temp_train_df
            result_test_df=temp_test_df
        else:
            result_train_df=pd.concat([result_train_df, temp_train_df],axis=1)
            result_test_df=pd.concat([result_test_df, temp_test_df],axis=1)
    
    return result_train_df, result_test_df


# let's run the frequency encoding function

# In[ ]:


cat_cols=['ps_ind_02_cat','ps_car_04_cat', 'ps_car_09_cat',
          'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_11_cat']

# generate dataframe for frequency features for the train and test dataset
train_freq, test_freq=freq_encoding(cat_cols, train, test)

# merge them into the original train and test dataset
train=pd.concat([train, train_freq], axis=1)
test=pd.concat([test,test_freq], axis=1)


# ## 1.2 Binary Encoding

# In[ ]:


# perform binary encoding for categorical variable
# this function take in a pair of train and test data set, and the feature that need to be encode.
# it returns the two dataset with input feature encoded in binary representation
# this function assumpt that the feature to be encoded is already been encoded in a numeric manner 
# ranging from 0 to n-1 (n = number of levels in the feature). 

def binary_encoding(train_df, test_df, feat):
    # calculate the highest numerical value used for numeric encoding
    train_feat_max = train_df[feat].max()
    test_feat_max = test_df[feat].max()
    if train_feat_max > test_feat_max:
        feat_max = train_feat_max
    else:
        feat_max = test_feat_max
        
    # use the value of feat_max+1 to represent missing value
    train_df.loc[train_df[feat] == -1, feat] = feat_max + 1
    test_df.loc[test_df[feat] == -1, feat] = feat_max + 1
    
    # create a union set of all possible values of the feature
    union_val = np.union1d(train_df[feat].unique(), test_df[feat].unique())

    # extract the highest value from from the feature in decimal format.
    max_dec = union_val.max()
    
    # work out how the ammount of digtis required to be represent max_dev in binary representation
    max_bin_len = len("{0:b}".format(max_dec))
    index = np.arange(len(union_val))
    columns = list([feat])
    
    # create a binary encoding feature dataframe to capture all the levels for the feature
    bin_df = pd.DataFrame(index=index, columns=columns)
    bin_df[feat] = union_val
    
    # capture the binary representation for each level of the feature 
    feat_bin = bin_df[feat].apply(lambda x: "{0:b}".format(x).zfill(max_bin_len))
    
    # split the binary representation into different bit of digits 
    splitted = feat_bin.apply(lambda x: pd.Series(list(x)).astype(np.uint8))
    splitted.columns = [feat + '_bin_' + str(x) for x in splitted.columns]
    bin_df = bin_df.join(splitted)
    
    # merge the binary feature encoding dataframe with the train and test dataset - Done! 
    train_df = pd.merge(train_df, bin_df, how='left', on=[feat])
    test_df = pd.merge(test_df, bin_df, how='left', on=[feat])
    return train_df, test_df


# let's run the binary encoding function

# In[ ]:


cat_cols=['ps_ind_02_cat','ps_car_04_cat', 'ps_car_09_cat',
          'ps_ind_05_cat', 'ps_car_01_cat']

train, test=binary_encoding(train, test, 'ps_ind_02_cat')
train, test=binary_encoding(train, test, 'ps_car_04_cat')
train, test=binary_encoding(train, test, 'ps_car_09_cat')
train, test=binary_encoding(train, test, 'ps_ind_05_cat')
train, test=binary_encoding(train, test, 'ps_car_01_cat')


# optionally, you can also choose to drop the original categorical features. Shoud you do it? I say trust your CV :)
# let's do this here jut for demonstration purpose

# In[ ]:


col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train.drop(col_to_drop, axis=1, inplace=True)  
test.drop(col_to_drop, axis=1, inplace=True)  


# ## 1.3 Feature Reduction
# Let's now drop all the features with the wording "cal" - "Cal, you are FIRED" ^ ^

# In[ ]:


col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train.drop(col_to_drop, axis=1, inplace=True)  
test.drop(col_to_drop, axis=1, inplace=True)


# Right, after the above data manipulation, we can now take a brief look at our dataset.

# In[ ]:


train.head(5)


# # 2. K-fold CV with Out-of-Fold Prediction
# 
# 
# *Note: for demonstration purpose, I have dump down the parameters of the models to make them run faster, so please take time to find a good cominbation of parameter when you send them out to battle for real!*
# 
# ## 2.1 OOF utility functions
# Firstly, let's write this handy function to convert AUC score into Gini Normalised Coeficient

# In[ ]:


def auc_to_gini_norm(auc_score):
    return 2*auc_score-1


# ### 2.1.1 Sklearn K-fold & OOF function
# Next up next provide a K-fold function that generate out-of-fold predictions for train and test data.

# In[ ]:


def cross_validate_sklearn(clf, x_train, y_train , x_test, kf,scale=False, verbose=True):
    start_time=time.time()
    
    # initialise the size of out-of-fold train an test prediction
    train_pred = np.zeros((x_train.shape[0]))
    test_pred = np.zeros((x_test.shape[0]))

    # use the kfold object to generate the required folds
    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        # generate training folds and validation fold
        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[test_index, :]
        y_train_kf, y_val_kf = y_train[train_index], y_train[test_index]

        # perform scaling if required i.e. for linear algorithms
        if scale:
            scaler = StandardScaler().fit(x_train_kf.values)
            x_train_kf_values = scaler.transform(x_train_kf.values)
            x_val_kf_values = scaler.transform(x_val_kf.values)
            x_test_values = scaler.transform(x_test.values)
        else:
            x_train_kf_values = x_train_kf.values
            x_val_kf_values = x_val_kf.values
            x_test_values = x_test.values
        
        # fit the input classifier and perform prediction.
        clf.fit(x_train_kf_values, y_train_kf.values)
        val_pred=clf.predict_proba(x_val_kf_values)[:,1]
        train_pred[test_index] += val_pred

        y_test_preds = clf.predict_proba(x_test_values)[:,1]
        test_pred += y_test_preds

        fold_auc = roc_auc_score(y_val_kf.values, val_pred)
        fold_gini_norm = auc_to_gini_norm(fold_auc)

        if verbose:
            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc, fold_gini_norm))

    test_pred /= kf.n_splits

    cv_auc = roc_auc_score(y_train, train_pred)
    cv_gini_norm = auc_to_gini_norm(cv_auc)
    cv_score = [cv_auc, cv_gini_norm]
    if verbose:
        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))
        end_time = time.time()
        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))
    return cv_score, train_pred,test_pred


# ### 2.1.2 Xgboost K-fold & OOF function
# In this part, we are going to use the native interface of XGB and LGB, so the following functions are tailor for this. For sure it would be easiler just to call the respective sklearn api, but the native interfaces provide some nice additional capability. For instance, the 'hist' option to use fast histogram in XGB is only available via the native interface as far as I know. 
# 
# Also, we need to provide the following function to convert probability into rank for these two OOF function. The needs to use normalised rank instead of predicted probabilities will become appearent later in this notebook :) 

# In[ ]:


def probability_to_rank(prediction, scaler=1):
    pred_df=pd.DataFrame(columns=['probability'])
    pred_df['probability']=prediction
    pred_df['rank']=pred_df['probability'].rank()/len(prediction)*scaler
    return pred_df['rank'].values


# The following is the k-fold function for XGB to generate OOF predictions, this function is very much similar to its sklearn counter part. The difference is that we need to use the XGB interface to facilitate the classifer, also we provide an option cover probability into rank.

# In[ ]:


def cross_validate_xgb(params, x_train, y_train, x_test, kf, cat_cols=[], verbose=True, 
                       verbose_eval=50, num_boost_round=4000, use_rank=True):
    start_time=time.time()

    train_pred = np.zeros((x_train.shape[0]))
    test_pred = np.zeros((x_test.shape[0]))

    # use the k-fold object to enumerate indexes for each training and validation fold
    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)): # folds 1, 2 ,3 ,4, 5
        # example: training from 1,2,3,4; validation from 5
        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[val_index, :]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]
        x_test_kf=x_test.copy()

        d_train_kf = xgb.DMatrix(x_train_kf, label=y_train_kf)
        d_val_kf = xgb.DMatrix(x_val_kf, label=y_val_kf)
        d_test = xgb.DMatrix(x_test_kf)

        bst = xgb.train(params, d_train_kf, num_boost_round=num_boost_round,
                        evals=[(d_train_kf, 'train'), (d_val_kf, 'val')], verbose_eval=verbose_eval,
                        early_stopping_rounds=50)

        val_pred = bst.predict(d_val_kf, ntree_limit=bst.best_ntree_limit)
        if use_rank:
            train_pred[val_index] += probability_to_rank(val_pred)
            test_pred+=probability_to_rank(bst.predict(d_test))
        else:
            train_pred[val_index] += val_pred
            test_pred+=bst.predict(d_test)

        fold_auc = roc_auc_score(y_val_kf.values, val_pred)
        fold_gini_norm = auc_to_gini_norm(fold_auc)

        if verbose:
            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc, 
                                                                                     fold_gini_norm))

    test_pred /= kf.n_splits

    cv_auc = roc_auc_score(y_train, train_pred)
    cv_gini_norm = auc_to_gini_norm(cv_auc)
    cv_score = [cv_auc, cv_gini_norm]
    if verbose:
        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))
        end_time = time.time()
        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))

        return cv_score, train_pred,test_pred


# ### 2.1.3 LigthGBM K-fold & OOF function
# The same function for LGB, this one is almost identifical to the one for XGB, apart from code that call the LightGBM interface

# In[ ]:


def cross_validate_lgb(params, x_train, y_train, x_test, kf, cat_cols=[],
                       verbose=True, verbose_eval=50, use_cat=True, use_rank=True):
    start_time = time.time()
    train_pred = np.zeros((x_train.shape[0]))
    test_pred = np.zeros((x_test.shape[0]))

    if len(cat_cols)==0: use_cat=False

    # use the k-fold object to enumerate indexes for each training and validation fold
    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)): # folds 1, 2 ,3 ,4, 5
        # example: training from 1,2,3,4; validation from 5
        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[val_index, :]
        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]

        if use_cat:
            lgb_train = lgb.Dataset(x_train_kf, y_train_kf, categorical_feature=cat_cols)
            lgb_val = lgb.Dataset(x_val_kf, y_val_kf, reference=lgb_train, categorical_feature=cat_cols)
        else:
            lgb_train = lgb.Dataset(x_train_kf, y_train_kf)
            lgb_val = lgb.Dataset(x_val_kf, y_val_kf, reference=lgb_train)

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=4000,
                        valid_sets=lgb_val,
                        early_stopping_rounds=30,
                        verbose_eval=verbose_eval)

        val_pred = gbm.predict(x_val_kf)

        if use_rank:
            train_pred[val_index] += probability_to_rank(val_pred)
            test_pred += probability_to_rank(gbm.predict(x_test))
            # test_pred += gbm.predict(x_test)
        else:
            train_pred[val_index] += val_pred
            test_pred += gbm.predict(x_test)

        # test_pred += gbm.predict(x_test)
        fold_auc = roc_auc_score(y_val_kf.values, val_pred)
        fold_gini_norm = auc_to_gini_norm(fold_auc)
        if verbose:
            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc, fold_gini_norm))

    test_pred /= kf.n_splits

    cv_auc = roc_auc_score(y_train, train_pred)
    cv_gini_norm = auc_to_gini_norm(cv_auc)
    cv_score = [cv_auc, cv_gini_norm]
    if verbose:
        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))
        end_time = time.time()
        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))
    return cv_score, train_pred,test_pred


# # 3. Generate level 1 OOF predictions
# Almost there to actually generate some level OOF output! last things to do is the prepare our train and test data for our dear machine learning algorithms, and create the StratifiedKFold object

# In[ ]:


drop_cols=['id','target']
y_train=train['target']
x_train=train.drop(drop_cols, axis=1)
x_test=test.drop(['id'], axis=1)


# Here, I would like remind you that for stacking, you SHALL use consistent fold distribution at ALL level for ALL your model. The technical reaons for this had been discussed at length in this [forum thread](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/43467)  by our right honourable fellow competitors

# In[ ]:


kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)


# Right! next generate some level 1 model output...

# ## 3.1 Random Forest
# Let's use the old good random forest algorithm

# In[ ]:


rf=RandomForestClassifier(n_estimators=200, n_jobs=6, min_samples_split=5, max_depth=7,
                          criterion='gini', random_state=0)

outcomes =cross_validate_sklearn(rf, x_train, y_train ,x_test, kf, scale=False, verbose=True)

rf_cv=outcomes[0]
rf_train_pred=outcomes[1]
rf_test_pred=outcomes[2]

rf_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=rf_train_pred)
rf_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=rf_test_pred)


# ## 3.2 Extra Tree
# We love tree! We love more trees, and therefore let's have extra tree :)

# In[ ]:


et=RandomForestClassifier(n_estimators=100, n_jobs=6, min_samples_split=5, max_depth=5,
                          criterion='gini', random_state=0)

outcomes =cross_validate_sklearn(et, x_train, y_train ,x_test, kf, scale=False, verbose=True)

et_cv=outcomes[0]
et_train_pred=outcomes[1]
et_test_pred=outcomes[2]

et_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=et_train_pred)
et_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=et_test_pred)


# ## 3.3 Logistic Regression
# Let's now throw in our favourite linear friend - Logistic Regression

# In[ ]:


logit=LogisticRegression(random_state=0, C=0.5)

outcomes = cross_validate_sklearn(logit, x_train, y_train ,x_test, kf, scale=True, verbose=True)

logit_cv=outcomes[0]
logit_train_pred=outcomes[1]
logit_test_pred=outcomes[2]

logit_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=logit_train_pred)
logit_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=logit_test_pred)


# ## 3.4 BernoulliNB
# A little bit of diversity from Naive Bayes never heard, this one of those algorithms that normally don't generate sigle output that rival XGB/LGB, but nevertheless help to improve the overal stacking performance due the diversity it bring to the party

# In[ ]:


nb=BernoulliNB()

outcomes =cross_validate_sklearn(nb, x_train, y_train ,x_test, kf, scale=True, verbose=True)

nb_cv=outcomes[0]
nb_train_pred=outcomes[1]
nb_test_pred=outcomes[2]

nb_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=nb_train_pred)
nb_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=nb_test_pred)


# ## 3.5 XGB
# Now this is our go-to GBM Bazooka:

# In[ ]:


xgb_params = {
    "booster"  :  "gbtree", 
    "objective"         :  "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "auc",
    "eta": 0.1,
    "max_depth": 5,
    "min_child_weight": 10,
    "gamma": 0.70,
    "subsample": 0.76,
    "colsample_bytree": 0.95,
    "nthread": 6,
    "seed": 0,
    'silent': 1
}

outcomes=cross_validate_xgb(xgb_params, x_train, y_train, x_test, kf, use_rank=False, verbose_eval=False)

xgb_cv=outcomes[0]
xgb_train_pred=outcomes[1]
xgb_test_pred=outcomes[2]

xgb_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=xgb_train_pred)
xgb_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=xgb_test_pred)


# ## 3.6 LightGBM
# There is a crack in everything, that's how the light gets in :) 

# In[ ]:


lgb_params = {
    'task': 'train',
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 22,
    'min_sum_hessian_in_leaf': 20,
    'max_depth': 5,
    'learning_rate': 0.1,  # 0.618580
    'num_threads': 6,
    'feature_fraction': 0.6894,
    'bagging_fraction': 0.4218,
    'max_drop': 5,
    'drop_rate': 0.0123,
    'min_data_in_leaf': 10,
    'bagging_freq': 1,
    'lambda_l1': 1,
    'lambda_l2': 0.01,
    'verbose': 1
}


cat_cols=['ps_ind_02_cat','ps_car_04_cat', 'ps_car_09_cat','ps_ind_05_cat', 'ps_car_01_cat']
outcomes=cross_validate_lgb(lgb_params,x_train, y_train ,x_test,kf, cat_cols, use_cat=True, 
                            verbose_eval=False, use_rank=False)

lgb_cv=outcomes[0]
lgb_train_pred=outcomes[1]
lgb_test_pred=outcomes[2]

lgb_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=lgb_train_pred)
lgb_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=lgb_test_pred)


# We now have our level 1 friends ready, lets proceed and send them into the stacking party!

# # 4. Level 2 ensemble

# ## 4.1 Generate L1 output dataframe
# Let's group ouf level 1 OOF predictions output together to genenerate the input for level 2 stacking

# In[ ]:


columns=['rf','et','logit','nb','xgb','lgb']
train_pred_df_list=[rf_train_pred_df, et_train_pred_df, logit_train_pred_df, nb_train_pred_df,
                    xgb_train_pred_df, lgb_train_pred_df]

test_pred_df_list=[rf_test_pred_df, et_test_pred_df, logit_test_pred_df, nb_test_pred_df,
                    xgb_test_pred_df, lgb_test_pred_df]

lv1_train_df=pd.DataFrame(columns=columns)
lv1_test_df=pd.DataFrame(columns=columns)

for i in range(0,len(columns)):
    lv1_train_df[columns[i]]=train_pred_df_list[i]['prediction_probability']
    lv1_test_df[columns[i]]=test_pred_df_list[i]['prediction_probability']



# ## 4.2 Level 2 XGB
# Back to XGB for level 2! everything shall be the same, paint old easy mdoel building, right? well..

# In[ ]:


xgb_lv2_outcomes=cross_validate_xgb(xgb_params, lv1_train_df, y_train, lv1_test_df, kf, 
                                          verbose=True, verbose_eval=False, use_rank=False)

xgb_lv2_cv=xgb_lv2_outcomes[0]
xgb_lv2_train_pred=xgb_lv2_outcomes[1]
xgb_lv2_test_pred=xgb_lv2_outcomes[2]


# So what just happened there?  Our CV score for each training fold is pretty descent, but our overall training CV score just fell through the crack!  Well, it turns out since we are using AUC/Gini as metric which is ranking dependent, and it turns out that if you apply xgb and lgb at level 2 stacking, the ranking get messed up when each fold's prediction scores are put together.  And this goes back to why we implemented that function to convert probability into ranks earlier.
# 
# Now, let's use the *use_rank* option, and see what happens:

# In[ ]:


xgb_lv2_outcomes=cross_validate_xgb(xgb_params, lv1_train_df, y_train, lv1_test_df, kf, 
                                          verbose=True, verbose_eval=False, use_rank=True)

xgb_lv2_cv=xgb_lv2_outcomes[0]
xgb_lv2_train_pred=xgb_lv2_outcomes[1]
xgb_lv2_test_pred=xgb_lv2_outcomes[2]


# Much better, the OOF score for train prediction looks great! and you can see the score here is better already than any of the level 1 OOF train score. The best score in level 1 comes from XGB with 0.282 region, and we are now on 0.284

#  ## 4.3 Level 2 LightGBM
# Same story for LightGBM at level 2, we need to use the *use_rank* option:

# In[ ]:


lgb_lv2_outcomes=cross_validate_lgb(lgb_params,lv1_train_df, y_train ,lv1_test_df,kf, [], use_cat=False, 
                                    verbose_eval=False, use_rank=True)

lgb_lv2_cv=xgb_lv2_outcomes[0]
lgb_lv2_train_pred=lgb_lv2_outcomes[1]
lgb_lv2_test_pred=lgb_lv2_outcomes[2]


# No surprise here :)

# ## 4.3 Level 2 Random Forest
# Now let's try a few more algorithms on level 2, and let's revisit random forest again.

# In[ ]:


rf_lv2=RandomForestClassifier(n_estimators=200, n_jobs=6, min_samples_split=5, max_depth=7,
                          criterion='gini', random_state=0)
rf_lv2_outcomes = cross_validate_sklearn(rf_lv2, lv1_train_df, y_train ,lv1_test_df, kf, 
                                            scale=True, verbose=True)
rf_lv2_cv=rf_lv2_outcomes[0]
rf_lv2_train_pred=rf_lv2_outcomes[1]
rf_lv2_test_pred=rf_lv2_outcomes[2]


# ## 4.4 Level 2 Logistic Regression
# Logistic Regression, take 2

# In[ ]:


logit_lv2=LogisticRegression(random_state=0, C=0.5)
logit_lv2_outcomes = cross_validate_sklearn(logit_lv2, lv1_train_df, y_train ,lv1_test_df, kf, 
                                            scale=True, verbose=True)
logit_lv2_cv=logit_lv2_outcomes[0]
logit_lv2_train_pred=logit_lv2_outcomes[1]
logit_lv2_test_pred=logit_lv2_outcomes[2]


# Hopefully by now you can see that on level 2, models like random forest and logistic regression are now producing very competivie results thanks to the meta-features from the level 1 OOF output.
# 
# We are having fun! and why stop in level 2? let's bring on level 3 :)

# # 5. Level 3 ensemble
# On level 3, we follow simlar workflow as level 2. First we put the OOF output from level 2 together, and then send them to our chosen algorithms.
# 
# ## 5.1 Generate L2 output dataframe

# In[ ]:


lv2_columns=['rf_lf2', 'logit_lv2', 'xgb_lv2','lgb_lv2']
train_lv2_pred_list=[rf_lv2_train_pred, logit_lv2_train_pred, xgb_lv2_train_pred, lgb_lv2_train_pred]

test_lv2_pred_list=[rf_lv2_test_pred, logit_lv2_test_pred, xgb_lv2_test_pred, lgb_lv2_test_pred]

lv2_train=pd.DataFrame(columns=lv2_columns)
lv2_test=pd.DataFrame(columns=lv2_columns)

for i in range(0,len(lv2_columns)):
    lv2_train[lv2_columns[i]]=train_lv2_pred_list[i]
    lv2_test[lv2_columns[i]]=test_lv2_pred_list[i]


# ## 5.2 Level 3 XGB 
# On this level, let's just stay with our trusted weapon XGB

# In[ ]:


xgb_lv3_params = {
    "booster"  :  "gbtree", 
    "objective"         :  "binary:logistic",
    "tree_method": "hist",
    "eval_metric": "auc",
    "eta": 0.1,
    "max_depth": 2,
    "min_child_weight": 10,
    "gamma": 0.70,
    "subsample": 0.76,
    "colsample_bytree": 0.95,
    "nthread": 6,
    "seed": 0,
    'silent': 1
}



xgb_lv3_outcomes=cross_validate_xgb(xgb_lv3_params, lv2_train, y_train, lv2_test, kf, 
                                          verbose=True, verbose_eval=False, use_rank=True)

xgb_lv3_cv=xgb_lv3_outcomes[0]
xgb_lv3_train_pred=xgb_lv3_outcomes[1]
xgb_lv3_test_pred=xgb_lv3_outcomes[2]


# This is slightly better than the XGB ouput at level 2, but not by much, as we are now seeing diminsing return as the level improve. Let's try tp pair this with something linear.

# ## 5.3 Level 3 Logistic Regression
# and of course that something linear is going to be Logistic Regression

# In[ ]:


logit_lv3=LogisticRegression(random_state=0, C=0.5)
logit_lv3_outcomes = cross_validate_sklearn(logit_lv3, lv2_train, y_train ,lv2_test, kf, 
                                            scale=True, verbose=True)
logit_lv3_cv=logit_lv3_outcomes[0]
logit_lv3_train_pred=logit_lv3_outcomes[1]
logit_lv3_test_pred=logit_lv3_outcomes[2]


# At this level, we don't see that much different between XGB and Logistic Regression anymore.

# ## 5.4 Average L3 outputs & Submission Generation

# We can always still do a simple weight average, to bring the two together and see if there any extra juice to be squeezed

# In[ ]:


weight_avg=logit_lv3_train_pred*0.5+ xgb_lv3_train_pred*0.5
print(auc_to_gini_norm(roc_auc_score(y_train, weight_avg)))


# Well, for training score, we manage to arravie at 0.28443.
# We can now try to apply the same weight distribution to generate our submission.

# In[ ]:


submission=sample_submission.copy()
submission['target']=logit_lv3_test_pred*0.5+ xgb_lv3_test_pred*0.5
filename='stacking_demonstration.csv.gz'
submission.to_csv(filename,compression='gzip', index=False)


# # 6 After Thought

# So I hope this three-level stacking guide is useful to demonstrate how you can capture more information from the training data, and hopefully this can generate to better test prediction. Well I use the word "hopefully" here as we all know the dataset in this competition is pretty noisy.   and in truth I am not sure if stacking beyong 2 level would bring much benefit, but then we will learn from the highflyers who survive the shake up!
# 
# My personal likely strategy to approach stacking is probably:
# * go with a 2-level approach, and weight average on level 2
# * applying the same stacking routine to several different random seed. 
# * weigh average the above
# 
# I seriously think robust CV is the key for this competition, and always be suspicious of all things shared on kernel. Alternatively, perhaps with 1 day to go, someone will share a leak or a 0.291 script to send us into a frenzy? One can always hope...
# 
# Enjoy every bit of these last days - May all the insured drivers never have to claim! :)
# 
