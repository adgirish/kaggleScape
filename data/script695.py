
# coding: utf-8

# Just a little look at how prices impact interest_level

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Some useful imports

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read data

# In[ ]:


train = pd.read_json('../input/train.json').set_index('listing_id')
test = pd.read_json('../input/test.json').set_index('listing_id')


# ## Scatter plot:  price vs bedrooms colored by interest_level 

# In[ ]:


# limit number of bedrooms and prices
usable_train =  train[(train.price<10000) & (train.bedrooms<=4)]
palette = {"high": "r", "low":"g", "medium":"orange"}
plt.figure(figsize=(11,10))
for interest in ['low', 'medium', 'high']:
    plt.scatter(usable_train[usable_train.interest_level==interest].bedrooms, 
                usable_train[usable_train.interest_level==interest].price, 
                c=palette[interest])


# High interest goes for low prices

# ## Boxplot gives even better insights 

# In[ ]:


plt.figure(figsize=(11,10))
sns.boxplot(x="bedrooms", y="price", hue="interest_level", data=usable_train, palette=palette)


# ## How can we use this ?

# In[ ]:


def add_median_price(key=None, suffix="", trn_df=None, tst_df=None):
    """
    Compute median prices for renthop dataset.
    The function adds 2 columns to the pandas DataFrames : the median prices and a ratio
    between nthe actual price of the rent and the median
    
    :param key: list of columns on which to groupby and compute median prices
    :param suffix: string used to suffix the newly created columns/features
    :param trn_df: training dataset as a pandas DataFrame
    :param tst_df: test dataset as a pandas DataFrame
    :return: updated train and test DataFrames

    :Example
    
    train, test = add_median_price(key=['bedrooms', 'bathrooms'], 
                                   suffix='rooms', 
                                   trn_df=train, 
                                   tst_df=test)

    """
    # Set features to be used
    median_features = key.copy()
    median_features.append('price')
    # Concat train and test to find median prices over whole dataset
    median_prices = pd.concat([trn_df[median_features], tst_df[median_features]], axis=0)
    # Group data by key to compute median prices
    medians_by_key = median_prices.groupby(by=key)['price'].median().reset_index()
    # Rename median column with provided suffix
    medians_by_key.rename(columns={'price': 'median_price_' + suffix}, inplace=True)
    # Update data frames, note that merge seems to reset the index
    # that's why I reset first and set again the index
    trn_df = trn_df.reset_index().merge(medians_by_key, on=key, how='left').set_index('listing_id')
    tst_df = tst_df.reset_index().merge(medians_by_key, on=key, how='left').set_index('listing_id')
    trn_df['price_to_median_ratio_' + suffix] = trn_df['price'] /trn_df['median_price_' + suffix]
    tst_df['price_to_median_ratio_' + suffix] = tst_df['price'] / tst_df['median_price_' + suffix]

    return trn_df, tst_df


# ## Define classifier over 10 folds

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
import time

def run_classifier():
    n_folds = 3
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=67594235)
    train_features = [ f for f in train.columns 
                      if (train[f].dtype != 'object') & (f != 'interest_level') ]

    target_num_map = {'high': 0, 'medium': 1, 'low': 2}
    target = np.array(train['interest_level'].apply(lambda x: target_num_map[x]))

    tst_scores = []
    tst_rounds = []

    data_X = train[train_features].values
    start = time.time()
    features_imp = np.zeros(len(train_features))
    for fold, (trn_idx, tst_idx) in enumerate(skf.split(data_X, target)):
        # Create a classifier
        clf = XGBClassifier(n_estimators=10000,
                            objective='multi:softprob',
                            learning_rate=0.3,
                            max_depth=3,
                            min_child_weight=1,
                            subsample=.8,
                            colsample_bytree=.9,
                            colsample_bylevel=.5,
                            gamma=0.0005,
                            scale_pos_weight=1,
                            base_score=.5,
                            reg_lambda=0,
                            reg_alpha=0,
                            missing=0,
                            seed=0)

        # Split the data
        trn_X = data_X[trn_idx]
        trn_Y = target[trn_idx]
        tst_X = data_X[tst_idx]
        tst_Y = target[tst_idx]

        # Train the model
        clf.fit(trn_X, trn_Y,
            eval_set=[(trn_X, trn_Y), (tst_X, tst_Y)],
            verbose=False,
            eval_metric='mlogloss',
            early_stopping_rounds=50)

        # Get features importance
        features_imp += clf.feature_importances_

        # Predict the data
        preds = clf.predict_proba(tst_X, ntree_limit=clf.best_ntree_limit)

        tst_scores.append(log_loss(tst_Y, preds))
        tst_rounds.append(clf.best_ntree_limit)
    
        print("LogLoss for fold %2d : %.5f" % (fold+1, log_loss(tst_Y, preds)))

    print("Average LogLoss : %.5f / %.6f in %4d rounds [%5.1f mn]"
          % (np.mean(tst_scores),
             np.std(tst_scores),
             np.mean(tst_rounds), (time.time() - start)/60))
    
    return train_features, features_imp


# ## Train with basic features

# In[ ]:


for df in (train, test):
    df['nb_images'] = df['photos'].apply(len)
    df['nb_features'] = df['features'].apply(len)
    df['nb_words'] = df['description'].apply(lambda x: len(x.split()))

train_features, features_imp = run_classifier()
                                             


# ## Train with median price over bedrooms

# In[ ]:


train, test = add_median_price(key=['bedrooms'],
                               suffix="bed",
                               trn_df=train, tst_df=test)

train_features, features_imp = run_classifier()


# ## Show feature_importances

# In[ ]:


imp_df = pd.DataFrame(data={'feature': train_features, 
                            'importance': features_imp})
imp_df.sort_values(by='importance', ascending=False, inplace=True)
sns.barplot(x="importance", y="feature", data=imp_df)

