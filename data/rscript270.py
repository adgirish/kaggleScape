"""
XGBoost on Hist mode with lossguide to give lightgbm characteristic
- Features from Andy Harless' script https://www.kaggle.com/aharless/try-pranav-s-r-lgbm-in-python
- Processing last 50 million observations in two chunks and then wa 
- Note that available disk space is 1 GB only so save chunk wise preds in gzip
- With this method, you can process entire training data in chunks without hitting memory limit. 
- And lastly, please excuse my not-so-efficient/ repititive code chunks in Python. 
"""

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance
import matplotlib.pyplot as plt
from scipy.special import expit, logit
import gc

"""
Part 1: First XGB with last 25 million observations
"""

print("processing first chunk of 25 mln observations...")

path = '../input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('loading train data...')

train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv(path+"train.csv", skiprows=range(1,159903891), nrows=25000000,dtype=dtypes, usecols=train_cols)
# total observations: 184,903,891

y = train_df['is_attributed']
train_df.drop(['is_attributed'], axis=1, inplace=True)

print('loading test data...')
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=test_cols)

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']
test_df.drop(['click_id'], axis=1, inplace=True)

len_train = len(train_df)
df=train_df.append(test_df)

del test_df
gc.collect()

print('Data preparation...')

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
df.drop(['click_time'], axis=1, inplace=True)
gc.collect()

df['in_test_hh'] = (   3 
                     - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                     - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')

print('group by : ip_day_test_hh')
gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day',
         'in_test_hh'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_day_test_hh'})
df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
del gp
df.drop(['in_test_hh'], axis=1, inplace=True)
df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
gc.collect()

print('group by : ip_day_hh')
gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_day_hh'})
df = df.merge(gp, on=['ip','day','hour'], how='left')
del gp
df['nip_day_hh'] = df['nip_day_hh'].astype('uint16')
gc.collect()

print('group by : ip_hh_os')
gp = df[['ip', 'day', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_os'})
df = df.merge(gp, on=['ip','os','hour','day'], how='left')
del gp
df['nip_hh_os'] = df['nip_hh_os'].astype('uint16')
gc.collect()

print('group by : ip_hh_app')
gp = df[['ip', 'app', 'hour', 'day', 'channel']].groupby(by=['ip', 'app', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_app'})
df = df.merge(gp, on=['ip','app','hour','day'], how='left')
del gp
df['nip_hh_app'] = df['nip_hh_app'].astype('uint16')
gc.collect()

print('group by : ip_hh_dev')
gp = df[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_dev'})
df = df.merge(gp, on=['ip','device','day','hour'], how='left')
del gp
df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint32')
gc.collect()

df.drop( ['ip','day'], axis=1, inplace=True )
gc.collect()
print( df.info() )


#split back
test  = df[len_train:]
train = df[:(len_train)]
del df
gc.collect()

print("train size: ", len(train))
print("test size : ", len(test))

start_time = time.time()

"""
XGBoost parameters tuning guide:
https://github.com/dmlc/xgboost/blob/master/doc/how_to/param_tuning.md
https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
"""
params = {'eta': 0.1, 
          'tree_method': "hist",      # Fast histogram optimized approximate greedy algorithm. 
          'grow_policy': "lossguide", # split at nodes with highest loss change
          'max_leaves': 1400,         # Maximum number of nodes to be added. (for lossguide grow policy)
          'max_depth': 0,             # 0 means no limit (useful only for depth wise grow policy)
          'subsample': 0.7,           
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,       # The larger, the more conservative the algorithm will be
          'alpha':4,                  # L1 reglrz. on weights | large value = more conservative model
          'objective': 'binary:logistic', 
          'scale_pos_weight':99.7,
          'eval_metric': 'auc', 
          'nthread':4,
          'random_state': 84, 
          'silent': True}

x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=84)

del train
gc.collect()

# watch list to observe the change in error in training and holdout data
watchlist = [(xgb.DMatrix(x2, y2), 'valid')]

model = xgb.train(params, xgb.DMatrix(x1, y1), 50, watchlist, maximize=True, early_stopping_rounds = 5, verbose_eval=10)

del x1, x2, y1, y2
gc.collect()

print('[{}]: Training time for 1st XGB'.format(time.time() - start_time))


print("predicting...")
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('sub_xgb1.csv',index=False)

#Model evaluation
print("Extract feature importance matrix")
plot_importance(model)
plt.gcf().savefig('xgb1_fe.png')
del(model)
print("part 1 finished...")

print("clear everything before processing second chunk..")

#I hope this is equivalent to R's rm( list = ls())

for name in dir():
 if not name.startswith('_'):
      del globals()[name]
del(name)


"""
Part 2: Second XGB with second last 25 million observations
"""
print("processing second chunk of 25 mln observations...")

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance
import matplotlib.pyplot as plt
from scipy.special import expit, logit
import gc

path = '../input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('loading train data...')

train_cols = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
train_df = pd.read_csv(path+"train.csv", skiprows=range(1,134903891), nrows=25000000,dtype=dtypes, usecols=train_cols)

y = train_df['is_attributed']
train_df.drop(['is_attributed'], axis=1, inplace=True)

print('loading test data...')
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=test_cols)

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']
test_df.drop(['click_id'], axis=1, inplace=True)

len_train = len(train_df)
df=train_df.append(test_df)

del test_df
gc.collect()

print('Data preparation...')

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
df.drop(['click_time'], axis=1, inplace=True)
gc.collect()

df['in_test_hh'] = (   3 
                     - 2*df['hour'].isin(  most_freq_hours_in_test_data ) 
                     - 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')

print('group by : ip_day_test_hh')
gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day',
         'in_test_hh'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_day_test_hh'})
df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
del gp
df.drop(['in_test_hh'], axis=1, inplace=True)
df['nip_day_test_hh'] = df['nip_day_test_hh'].astype('uint32')
gc.collect()

print('group by : ip_day_hh')
gp = df[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_day_hh'})
df = df.merge(gp, on=['ip','day','hour'], how='left')
del gp
df['nip_day_hh'] = df['nip_day_hh'].astype('uint16')
gc.collect()

print('group by : ip_hh_os')
gp = df[['ip', 'day', 'os', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_os'})
df = df.merge(gp, on=['ip','os','hour','day'], how='left')
del gp
df['nip_hh_os'] = df['nip_hh_os'].astype('uint16')
gc.collect()

print('group by : ip_hh_app')
gp = df[['ip', 'app', 'hour', 'day', 'channel']].groupby(by=['ip', 'app', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_app'})
df = df.merge(gp, on=['ip','app','hour','day'], how='left')
del gp
df['nip_hh_app'] = df['nip_hh_app'].astype('uint16')
gc.collect()

print('group by : ip_hh_dev')
gp = df[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
         'hour'])[['channel']].count().reset_index().rename(index=str, 
         columns={'channel': 'nip_hh_dev'})
df = df.merge(gp, on=['ip','device','day','hour'], how='left')
del gp
df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint32')
gc.collect()

df.drop( ['ip','day'], axis=1, inplace=True )
gc.collect()
print( df.info() )


#split back
test  = df[len_train:]
train = df[:(len_train)]
del df
gc.collect()

print("train size: ", len(train))
print("test size : ", len(test))

start_time = time.time()

"""
XGBoost parameters tuning guide:
https://github.com/dmlc/xgboost/blob/master/doc/how_to/param_tuning.md
https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
"""
params = {'eta': 0.1, 
          'tree_method': "hist",      # Fast histogram optimized approximate greedy algorithm. 
          'grow_policy': "lossguide", # split at nodes with highest loss change
          'max_leaves': 1400,         # Maximum number of nodes to be added. (for lossguide grow policy)
          'max_depth': 0,             # 0 means no limit (useful only for depth wise grow policy)
          'subsample': 0.7,           
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,       # The larger, the more conservative the algorithm will be
          'alpha':4,                  # L1 reglrz. on weights | large value = more conservative model
          'objective': 'binary:logistic', 
          'scale_pos_weight':99.7,
          'eval_metric': 'auc', 
          'nthread':4,
          'random_state': 84, 
          'silent': True}

x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=84)

del train
gc.collect()

# watch list to observe the change in error in training and holdout data
watchlist = [(xgb.DMatrix(x2, y2), 'valid')]

model = xgb.train(params, xgb.DMatrix(x1, y1), 50, watchlist, maximize=True, early_stopping_rounds = 10, verbose_eval=10)

del x1, x2, y1, y2
gc.collect()

print('[{}]: Training time for 2nd XGB'.format(time.time() - start_time))


print("predicting...")
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('sub_xgb2.csv',index=False)

#Model evaluation
print("Extract feature importance matrix")
plot_importance(model)
plt.gcf().savefig('xgb2_fe.png')

print("part 2 finished...")

print("clear everything before processing third part..")
for name in dir():
 if not name.startswith('_'):
      del globals()[name]
del(name)


"""
Part 3: Average of both models for total 50 mln observations
"""

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance
import matplotlib.pyplot as plt
from scipy.special import expit, logit
import gc

print("part 3...")

almost_zero = 1e-10
almost_one = 1 - almost_zero

xgb1 = pd.read_csv("sub_xgb1.csv")
xgb2 = pd.read_csv("sub_xgb2.csv") 

final_sub = pd.DataFrame()
final_sub['click_id'] = xgb1['click_id']

final_sub['is_attributed'] = (  (xgb1['is_attributed'].clip(almost_zero,almost_one).apply(logit) * 0.5) +
                                (xgb2['is_attributed'].clip(almost_zero,almost_one).apply(logit) * 0.5)).apply(expit)

final_sub.to_csv("sub_xgb_wa_50mln.csv.gz", index=False,compression='gzip')

print("All done....")

