
# coding: utf-8

# This is feature elimination based on **[Boruta](https://m2.icm.edu.pl/boruta/)**. Thanks to **[@olivier](https://www.kaggle.com/ogrellier)** for **[discussions](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41595#233852)** and for **[this notebook](https://www.kaggle.com/ogrellier/noise-analysis-of-porto-seguro-s-features)** that got me going in this direction.
# 
# Note that olivier used LightGBM as his **[base estimator](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41595#234273)** while I am using Random Forest. Because of that, the results are different. I have no way of telling which feature selection is better as I haven't tested either one yet. If you do test them, please leave a note here. I will do the same.

# In[ ]:


from __future__ import print_function

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


# Loading files.

# In[ ]:


train = pd.read_csv('../input/train.csv', dtype={'target': np.int8, 'id': np.int32})
X = train.drop(['id','target'], axis=1).values
y = train['target'].values
tr_ids = train['id'].values
n_train = len(X)
test = pd.read_csv('../input/test.csv', dtype={'id': np.int32})
X_test = test.drop(['id'], axis=1).values
te_ids = test['id'].values


# It is worth playing with **[RFC parameters](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)**. Initially, I had *n_estimators=100* and *max_depth=10* which was not selecting enough features. Boruta parameters are explained **[here](https://github.com/scikit-learn-contrib/boruta_py)**.

# In[ ]:


rfc = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced', max_depth=6)
boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
start_time = timer(None)
boruta_selector.fit(X, y)
timer(start_time)


# The summary of the whole run is shown here. Couple of attributes at the end are commented out. Finally, we save train and test datasets with a subset of selected features.

# In[ ]:


print ('\n Initial features: ', train.drop(['id','target'], axis=1).columns.tolist() )

# number of selected features
print ('\n Number of selected features:')
print (boruta_selector.n_features_)

feature_df = pd.DataFrame(train.drop(['id','target'], axis=1).columns.tolist(), columns=['features'])
feature_df['rank']=boruta_selector.ranking_
feature_df = feature_df.sort_values('rank', ascending=True).reset_index(drop=True)
print ('\n Top %d features:' % boruta_selector.n_features_)
print (feature_df.head(boruta_selector.n_features_))
feature_df.to_csv('boruta-feature-ranking.csv', index=False)

# check ranking of features
print ('\n Feature ranking:')
print (boruta_selector.ranking_)

# check selected features
# print ('\n Selected features:')
# print (boruta_selector.support_)

# check weak features
# print ('\n Support for weak features:')
#print (boruta_selector.support_weak_)

selected = train.drop(['id','target'], axis=1).columns[boruta_selector.support_]
train = train[selected]
train['id'] = tr_ids
train['target'] = y
train = train.set_index('id')
train.to_csv('train_boruta_filtered.csv', index_label='id')
test = test[selected]
test['id'] = te_ids
test = test.set_index('id')
test.to_csv('test_boruta_filtered.csv', index_label='id')

