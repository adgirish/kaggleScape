
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tqdm import tqdm


# > ## Data loading, we parse transactiondate

# In[ ]:


train_df = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
test_df = pd.read_csv('../input/sample_submission.csv', low_memory=False)
properties = pd.read_csv('../input/properties_2016.csv', low_memory=False)
# field is named differently in submission
test_df['parcelid'] = test_df['ParcelId']


# In[ ]:


# similar to the1owl
def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df


# In[ ]:


train_df = add_date_features(train_df)
train_df = train_df.merge(properties, how='left', on='parcelid')
test_df = test_df.merge(properties, how='left', on='parcelid')
print("Train: ", train_df.shape)
print("Test: ", test_df.shape)


# # 0.a) Remove missing data fields

# In[ ]:


missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % exclude_missing)
print(len(exclude_missing))


# # 0.b) Remove data that is always the same

# In[ ]:


# exclude where we only have one unique value :D
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % exclude_unique)
print(len(exclude_unique))


# # 1.a) Define training features

# In[ ]:


exclude_other = ['parcelid', 'logerror']  # for indexing/training only
# do not know what this is LARS, 'SHCG' 'COR2YY' 'LNR2RPD-R3' ?!?
exclude_other.append('propertyzoningdesc')
train_features = []
for c in train_df.columns:
    if c not in exclude_missing        and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % train_features)
print(len(train_features))


# # 1.b) Define which of these training features are categorical

# In[ ]:


cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh        and not 'sqft' in c        and not 'cnt' in c        and not 'nbr' in c        and not 'number' in c:
        cat_feature_inds.append(i)
        
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])


# # 1.c) Fill missing values

# In[ ]:


# some out of range int is a good choice
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)


# # 2.a) Training time!

# In[ ]:


X_train = train_df[train_features]
y_train = train_df.logerror
print(X_train.shape, y_train.shape)


# In[ ]:


test_df['transactiondate'] = pd.Timestamp('2016-12-01')  # Dummy
test_df = add_date_features(test_df)
X_test = test_df[train_features]
print(X_test.shape)


# In[ ]:


num_ensembles = 5
y_pred = 0.0
for i in tqdm(range(num_ensembles)):
    # TODO(you): Use CV, tune hyperparameters
    model = CatBoostRegressor(
        iterations=200, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)
    model.fit(
        X_train, y_train,
        cat_features=cat_feature_inds)
    y_pred += model.predict(X_test)
y_pred /= num_ensembles


# # 3.) Create submission

# In[ ]:


submission = pd.DataFrame({
    'ParcelId': test_df['parcelid'],
})
# https://www.kaggle.com/c/zillow-prize-1/discussion/33899, Oct,Nov,Dec
test_dates = {
    '201610': pd.Timestamp('2016-09-30'),
    '201611': pd.Timestamp('2016-10-31'),
    '201612': pd.Timestamp('2016-11-30'),
    '201710': pd.Timestamp('2017-09-30'),
    '201711': pd.Timestamp('2017-10-31'),
    '201712': pd.Timestamp('2017-11-30')
}
for label, test_date in test_dates.items():
    print("Predicting for: %s ... " % (label))
    # TODO(you): predict for every `test_date`
    submission[label] = y_pred

submission_major = 1
submission.to_csv(
    'submission_%03d.csv' % (submission_major),
    float_format='%.4f',
    index=False)
print("Done! Good luck with submission #%d :)" % submission_major)

