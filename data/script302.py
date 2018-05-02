
# coding: utf-8

# # Feature Engineering & Importance Testing
# Goal of this notebook is to get an overview of all the features found around different kernels in this competition, as well as add a few new ones, and test their importance for predicting is_attributed using xgBoost. 
# 
# ## Current best score with these features
# Current I've got a public LB score of **0.9769** using V2 of these features, and the xgBoost parameters found by [Bayesian Optimization in this notebook](https://www.kaggle.com/nanomathias/bayesian-tuning-of-xgboost-lightgbm-lb-0-9769), training on the entire dataset.
# 
# ## Inspirational Notebooks
# Inspiration for features from (let me know if I have forgotten to give credit to anyone)
# * https://www.kaggle.com/nuhsikander/lgbm-new-features-corrected
# * https://www.kaggle.com/rteja1113/lightgbm-with-count-features
# * https://www.kaggle.com/aharless/swetha-s-xgboost-revised
# * https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977
# 
# 
# # 1. Loading data
# I'll just load a small subset of the data for more speedy testing

# In[4]:


import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load subset of the training data
X_train = pd.read_csv('../input/train.csv', nrows=1000000, parse_dates=['click_time'])

# Show the head of the table
X_train.head()


# # 2. Creating Features
# ## 2.1 Extracting time information
# First extract day, minute, hour, second from the click_time. 

# In[5]:


X_train['day'] = X_train['click_time'].dt.day.astype('uint8')
X_train['hour'] = X_train['click_time'].dt.hour.astype('uint8')
X_train['minute'] = X_train['click_time'].dt.minute.astype('uint8')
X_train['second'] = X_train['click_time'].dt.second.astype('uint8')
X_train.head()


# ## 2.2. Confidence Rates for is_attributed
# My thought is that some ips, apps, devices, etc. might have higher frequencies of is_attributed, and I wish to add that information, i.e. I'm calculating the following "attributed rates":
# 
# \begin{equation}
# \text{P}\,\text{(is_attributed}\,\,|\,\,\text{category)}
# \end{equation}
# 
# or in some cases two- or multiple-paired combinations:
# 
# \begin{equation}
# \text{P}\,\text{(is_attributed}\,\,|\,\,\text{category_1, category_2)}
# \end{equation}
# 
# The danger of this is that if a given category-combination has very few clicks, then the statistical significance of above equations cannot be trusted. Therefore I'll be weighing the rates by the following confidence rates:
# 
# \begin{equation}
#     \text{conf}_{\text{is_attributed}} = \frac{\log(\text{views}_{\text{category_1}})}{\log(100000)}
# \end{equation}
# 
# where the value 100000 has been chosen arbitrarily to such that if a given category has 1000 views, then it gets a confidence weight of 60%, if it has 100 views then onfly a confidence weight of 40% etc.

# In[6]:


ATTRIBUTION_CATEGORIES = [        
    # V1 Features #
    ###############
    ['ip'], ['app'], ['device'], ['os'], ['channel'],
    
    # V2 Features #
    ###############
    ['app', 'channel'],
    ['app', 'os'],
    ['app', 'device'],
    
    # V3 Features #
    ###############
    ['channel', 'os'],
    ['channel', 'device'],
    ['os', 'device']
]


# Find frequency of is_attributed for each unique value in column
freqs = {}
for cols in ATTRIBUTION_CATEGORIES:
    
    # New feature name
    new_feature = '_'.join(cols)+'_confRate'    
    
    # Perform the groupby
    group_object = X_train.groupby(cols)
    
    # Group sizes    
    group_sizes = group_object.size()
    log_group = np.log(100000) # 1000 views -> 60% confidence, 100 views -> 40% confidence 
    print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
        cols, new_feature, 
        group_sizes.max(), 
        np.round(group_sizes.mean(), 2),
        np.round(group_sizes.median(), 2),
        group_sizes.min()
    ))
    
    # Aggregation function
    def rate_calculation(x):
        """Calculate the attributed rate. Scale by confidence"""
        rate = x.sum() / float(x.count())
        conf = np.min([1, np.log(x.count()) / log_group])
        return rate * conf
    
    # Perform the merge
    X_train = X_train.merge(
        group_object['is_attributed']. \
            apply(rate_calculation). \
            reset_index(). \
            rename( 
                index=str,
                columns={'is_attributed': new_feature}
            )[cols + [new_feature]],
        on=cols, how='left'
    )
    
X_train.head()


# Locally I computed 10-fold cross-validation scores with 10million samples using xgBoost to see how each feature improved the model or not. The result was as follows:
#     
# <img src='http://i66.tinypic.com/120h01w.png' alt="Local Frequency CV tests" width='50%'/>
# 
# On the public LB the score was improved from **0.955 -> 0.9624** by just including these features

# ## 2.3. Group-By-Aggregation
# There are a lot of groupby -> count()/var()/mean() etc. feature engineering in the kernels I've checked out, so of course those have to be added as well :)

# In[8]:


# Define all the groupby transformations
GROUPBY_AGGREGATIONS = [
    
    # V1 - GroupBy Features #
    #########################    
    # Variance in day, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    {'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
    # Count, for ip-day-hour
    {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 
    
    # V2 - GroupBy Features #
    #########################
    # Average clicks on app by distinct users; is it an app they return to?
    {'groupby': ['app'], 
     'select': 'ip', 
     'agg': lambda x: float(len(x)) / len(x.unique()), 
     'agg_name': 'AvgViewPerDistinct'
    },
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},
    
    # V3 - GroupBy Features                                              #
    # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
    ###################################################################### 
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'}, 
    {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'}, 
    {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'}, 
    {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}, 
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'}    
]

# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    
    # Info
    print("Grouping by {}, and aggregating {} with {}".format(
        spec['groupby'], spec['select'], agg_name
    ))
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    
    # Perform the groupby
    gp = X_train[all_features].         groupby(spec['groupby'])[spec['select']].         agg(spec['agg']).         reset_index().         rename(index=str, columns={spec['select']: new_feature})
        
    # Merge back to X_total
    if 'cumcount' == spec['agg']:
        X_train[new_feature] = gp[0].values
    else:
        X_train = X_train.merge(gp, on=spec['groupby'], how='left')
        
     # Clear memory
    del gp
    gc.collect()

X_train.head()


# 1. Locally I computed 10-fold cross-validation scores with 10million samples using xgBoost to see how each feature improved the model or not. The result was as follows:
#     
# <img src='http://i68.tinypic.com/33xkis8.png' alt="Local groupby CV tests" width='50%'/>
# 
# On the public leaderboard the score was improved from **0.955 -> 0.9584** by just including these features

# # 2.4. Time till next click
# It might be interesting to know e.g. how long it takes for a given ip-app-channel before they perform the next click. So I'll create some features for these as well. ****

# In[ ]:


GROUP_BY_NEXT_CLICKS = [
    
    # V1
    {'groupby': ['ip']},
    {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    {'groupby': ['ip', 'os']},
    
    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']}
]

# Calculate the time to next click for each group
for spec in GROUP_BY_NEXT_CLICKS:
    
    # Name of new feature
    new_feature = '{}_nextClick'.format('_'.join(spec['groupby']))    
    
    # Unique list of features to select
    all_features = spec['groupby'] + ['click_time']
    
    # Run calculation
    print(f">> Grouping by {spec['groupby']}, and saving time to next click in: {new_feature}")
    X_train[new_feature] = X_train[all_features].groupby(spec['groupby']).click_time.transform(lambda x: x.diff().shift(-1)).dt.seconds
    
X_train.head()


# Locally I computed 10-fold cross-validation scores with 10million samples using xgBoost to see how each feature improved the model or not. The result was as follows:
#     
# <img src='http://i67.tinypic.com/2dt7y2p.png' alt="Local nextClick CV tests" width='50%'/>
# 
# On the public leaderboard the score was improved from **0.955 -> 0.9608** by just including these features

# ## 2.5. Clicks on app ad before & after
# Has the user previously or subsequently clicked the exact same app-device-os-channel? I thought that might be an interesting feature to test out as well.

# In[ ]:


HISTORY_CLICKS = {
    'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
    'app_clicks': ['ip', 'app']
}

# Go through different group-by combinations
for fname, fset in HISTORY_CLICKS.items():
    
    # Clicks in the past
    X_train['prev_'+fname] = X_train.         groupby(fset).         cumcount().         rename('prev_'+fname)
        
    # Clicks in the future
    X_train['future_'+fname] = X_train.iloc[::-1].         groupby(fset).         cumcount().         rename('future_'+fname).iloc[::-1]

# Count cumulative subsequent clicks
X_train.head()


# Locally I computed 10-fold cross-validation scores with 10million samples using xgBoost to see how each feature improved the model or not. The result was as follows:
#     
# <img src='http://i64.tinypic.com/2j675fr.png' alt="Local nextClick CV tests" width='50%'/>
# 
# Possibly a very small improvement. This small improvement was also seen on public LB, where the score went from **0.955 -> 0.9568** by just including these features

# # 3. Evaluating Feature Importance
# Having created heaps of features, I'll fit xgBoost to the data, and evaluate the feature importances. First split into X and y

# In[ ]:


import xgboost as xgb

# Split into X and y
y = X_train['is_attributed']
X = X_train.drop('is_attributed', axis=1).select_dtypes(include=[np.number])

# Create a model
# Params from: https://www.kaggle.com/aharless/swetha-s-xgboost-revised
clf_xgBoost = xgb.XGBClassifier(
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.7,
    colsample_bylevel = 0.7,
    scale_pos_weight = 9,
    min_child_weight = 0,
    reg_alpha = 4,
    n_jobs = 4, 
    objective = 'binary:logistic'
)
# Fit the models
clf_xgBoost.fit(X, y)


# The feature importances are MinMax scaled, put into a DataFrame, and finally plotted ordered by the mean feature importance.

# In[ ]:


from sklearn import preprocessing

# Get xgBoost importances
importance_dict = {}
for import_type in ['weight', 'gain', 'cover']:
    importance_dict['xgBoost-'+import_type] = clf_xgBoost.get_booster().get_score(importance_type=import_type)
    
# MinMax scale all importances
importance_df = pd.DataFrame(importance_dict).fillna(0)
importance_df = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(importance_df),
    columns=importance_df.columns,
    index=importance_df.index
)

# Create mean column
importance_df['mean'] = importance_df.mean(axis=1)

# Plot the feature importances
importance_df.sort_values('mean').plot(kind='bar', figsize=(20, 7))


# # Evaluation on Public Leaderboard
# I've evaluated the impact of all the above features on the public leaderboard - the results are as follows for the simple xgBoost model used in this notebook applyed to the entire training set, without any additional tuning etc.
# 
# <img src='http://i63.tinypic.com/jjtpow.png' width='50%' />
