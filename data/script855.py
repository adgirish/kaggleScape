
# coding: utf-8

# **Microsoft [LightGBM][1]** is a powerful, open-source boosted decision tree library similar to xgboost. In practice, it runs even faster than xgboost and achieves better performance in some cases.
# 
# To install LightGBM, follow the [installation guide][2] to get the C++ distribution. The python API can then be easily built with these [instructions][3].
# 
# Some useful resources for LightGBM python API and parameter tuning:
# 
# **[Python API Documentation][4]:** this page includes all the functions and objects
# 
# **[List of Parameters][5]:** all possible parameters for LightGBM functions and classes
# 
# **[Parameter Tuning Guide][6]:** the advanced parameter tuning guide for LightGBM. Since most parameters in LightGBM are similar to those in XGBoost, it should be intuitive to follow.
# 
# 
#   [1]: https://github.com/Microsoft/LightGBM
#   [2]: https://github.com/Microsoft/LightGBM/wiki/Installation-Guide
#   [3]: https://github.com/Microsoft/LightGBM/tree/master/python-package
#   [4]: https://github.com/Microsoft/LightGBM/blob/master/docs/Python-API.md
#   [5]: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.md
#   [6]: https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters-tuning.md

# ## Import Libraries, Preprocessing ##

# In[ ]:


import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import lightgbm as lgbm

def preprocess_1(train_df, test_df):
    """Just a generic preprocessing function, feel free to substitute it with your custom function"""
    # encode target variable
    train_df['interest_level'] = train_df['interest_level'].apply(lambda x: {'high': 0, 'medium': 1, 'low': 2}[x])
    test_df['interest_level'] = -1
    train_index = train_df.index
    test_index = test_df.index
    data_df = pd.concat((train_df, test_df), axis=0)
    del train_df, test_df
    
    # add counting features
    data_df['num_photos'] = data_df['photos'].apply(len)
    data_df['num_features'] = data_df['features'].apply(len)
    data_df['num_description'] = data_df['description'].apply(lambda x: len(x.split(' ')))
    data_df.drop('photos', axis=1, inplace=True)
    
    # naive feature engineering
    data_df['room_difference'] = data_df['bedrooms'] - data_df['bathrooms']
    data_df['total_rooms'] = data_df['bedrooms'] + data_df['bathrooms']
    data_df['price_per_room'] = data_df['price'] / (data_df['total_rooms'] + 1)
    
    # add datetime features
    data_df['created'] = pd.to_datetime(data_df['created'])
    data_df['c_month'] = data_df['created'].dt.month
    data_df['c_day'] = data_df['created'].dt.day
    data_df['c_hour'] = data_df['created'].dt.hour
    data_df['c_dayofyear'] = data_df['created'].dt.dayofyear
    data_df.drop('created', axis=1, inplace=True)
    
    # encode categorical features
    for col in ['display_address', 'street_address', 'manager_id', 'building_id']:
        data_df[col] = LabelEncoder().fit_transform(data_df[col])
       
    data_df.drop('description', axis=1, inplace=True)
    
    # get text features
    data_df['features'] = data_df['features'].apply(lambda x: ' '.join(['_'.join(i.split(' ')) for i in x]))
    textcv = CountVectorizer(stop_words='english', max_features=200)
    text_features = pd.DataFrame(textcv.fit_transform(data_df['features']).toarray(),
                                 columns=['f_' + format(x, '03d') for x in range(1, 201)],
                                 index=data_df.index)
    data_df = pd.concat(objs=(data_df, text_features), axis=1)
    data_df.drop('features', axis=1, inplace=True)
    
    feature_cols = [x for x in data_df.columns if x not in {'interest_level'}]
    return data_df.loc[train_index, feature_cols], data_df.loc[train_index, 'interest_level'],        data_df.loc[test_index, feature_cols]
    


# ## Load Data ##

# In[ ]:


train = pd.read_json(open("../input/train.json", "r"))
test = pd.read_json(open("../input/test.json", "r"))


# ## Define Hyperparameters for LightGBMClassifier ##

# In[ ]:


# the following dictionary contains most of the relavant hyperparameters for our task
# I haven't tuned them yet, so they are mostly default
t4_params = {
    'boosting_type': 'gbdt', 'objective': 'multiclass', 'nthread': -1, 'silent': True,
    'num_leaves': 2**4, 'learning_rate': 0.05, 'max_depth': -1,
    'max_bin': 255, 'subsample_for_bin': 50000,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.6, 'reg_alpha': 1, 'reg_lambda': 0,
    'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 10, 'scale_pos_weight': 1}

# they can be used directly to build a LGBMClassifier (which is wrapped in a sklearn fashion)
t4 = lgbm.sklearn.LGBMClassifier(n_estimators=1000, seed=0, **t4_params)


# ## Early Stopping with Cross Validation ##
# Similar to xgboost, we can use cross validation with early stopping to efficiently determine the optimal "**n_estimators**" value.

# In[ ]:


def cross_validate_lgbm(filename_str, preprocess_func=preprocess_1):
    lgbm_params = t4_params.copy()
    lgbm_params['num_class'] = 3
    train_X, train_y, test_df = preprocess_func(train, test)
    dset = lgbm.Dataset(train_X, train_y, silent=True)
    cv_results = lgbm.cv(
        lgbm_params, dset, num_boost_round=10000, nfold=5, stratified=False, shuffle=True, metrics='multi_logloss',
        early_stopping_rounds=100, verbose_eval=50, show_stdv=True, seed=0)
    # note: cv_results will look like: {"multi_logloss-mean": <a list of historical mean>,
    # "multi_logloss-stdv": <a list of historical standard deviation>}
    json.dump(cv_results, open(filename_str, 'w'))
    print(filename_str)
    print('best n_estimators:', len(cv_results['multi_logloss-mean']))
    print('best cv score:', cv_results['multi_logloss-mean'][-1])

# we simply have to run the following code each time we modify the hyperparameters:
cross_validate_lgbm('lgbm_1.json')

