
# coding: utf-8

# In[1]:


#It is just a start, i hope. I hope, I will have time for something more clever :-)
import os
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import lightgbm as lgb

print(os.getcwd())

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

NROWS1     = 3000000
NROWS2     = 1000000
RG         = 500000 #154903891
SKIPROWS   = range(1,RG)
path       = '../input/' 
path_train = path + 'train.csv'
path_test  =  path + 'test.csv'
train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols  = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('Loading the training data...')
train = pd.read_csv(path_train, skiprows=SKIPROWS, nrows=NROWS1, dtype=dtypes, header=0)
print('End loading train data...')
print('Loading the test data...')
test = pd.read_csv(path_test, dtype=dtypes, header=0)  #nrows=NROWS2,
print('End loading test data...')


# In[4]:


print("Train head")
print(train.head(5))
print("Test head")
print(test.head(5))


# In[6]:


print("Unique data train")
Unique_data_train = pd.to_datetime(train.click_time).dt.day.astype('uint8').value_counts().sort_index()
print(Unique_data_train)
print("Unique data test")
Unique_data_test =  pd.to_datetime(test.click_time).dt.day.astype('uint8').value_counts().sort_index()
print(Unique_data_test)


# In[7]:


len_train = len(train)
print('The initial size of the train set is', len_train)
train = train.append(test)
print('Binding the training and test set together...')
del test


# In[9]:


print("Train and test together")
print(train.head(5))


# In[11]:


#time
train['hour']    = pd.to_datetime(train.click_time).dt.hour.astype('uint8')
train['day']     = pd.to_datetime(train.click_time).dt.day.astype('uint8')
train['wday']    = pd.to_datetime(train.click_time).dt.dayofweek.astype('uint8')
train['minute']  = pd.to_datetime(train.click_time).dt.minute.astype('uint8')
train['second']  = pd.to_datetime(train.click_time).dt.second.astype('uint8')
train["doy"]     = pd.to_datetime(train.click_time).dt.dayofyear.astype('uint8')
train            = train.drop(['click_time', 'attributed_time'], axis=1)
print(train.dtypes) 


# In[12]:


print("Frequent hours")
frequent_hour = train.hour.value_counts().sort_index()
print(frequent_hour)
print("Frequent days")
frequent_day = train.day.value_counts().sort_index()
print(frequent_day)
print("Frequent doy")
frequent_doy = train.doy.value_counts().sort_index()
print(frequent_doy)
print("Frequent week days")
frequent_wday = train.wday.value_counts().sort_index()
print(frequent_wday)
print("Frequent minutes")
frequent_minute = train.minute.value_counts().sort_index()
print(frequent_minute)
print("Frequent seconds")
frequent_second = train.second.value_counts().sort_index()
print(frequent_minute)


# In[13]:


plt.figure(figsize=(15,20))

plt.subplot(321)
frequent_hour.plot(kind='bar')
plt.title("Frequent hours")
plt.xlabel("Hours")
plt.ylabel("Number")

plt.subplot(322)
frequent_day.plot(kind='bar')
plt.title("Frequent days")
plt.xlabel("Days")
plt.ylabel("Number")

plt.subplot(323)
frequent_doy.plot(kind='bar')
plt.title("Frequent day of year")
plt.xlabel("Frequent day of year")
plt.ylabel("Number")

plt.subplot(324)
frequent_wday.plot(kind='bar')
plt.title("Frequent day of week")
plt.xlabel("Frequent day of week")
plt.ylabel("Number")

plt.subplot(325)
frequent_minute.plot(kind='bar')
plt.xticks(np.arange(0, 69, step=10), (0,9,19,29,39,49,59))
plt.title("Frequent  minutes")
plt.xlabel("Minutes")
plt.ylabel("Number")

plt.subplot(326)
frequent_second.plot(kind='bar')
plt.xticks(np.arange(0, 69, step=10), (0,9,19,29,39,49,59))
plt.title("Frequent seconds")
plt.xlabel("Seconds")
plt.ylabel("Number")

del frequent_hour,frequent_day,frequent_doy,frequent_wday,frequent_minute,frequent_second
gc.collect()


# In[14]:


BIN = 30
plt.figure(figsize=(15,20))
plt.subplot(321)
plt.hist(train['hour'], bins=BIN)
plt.title("Histogram of frequent hours")
plt.xlabel("Frequent hours")

plt.subplot(322)
plt.hist(train['day'], bins=BIN)
plt.title("Histogram of frequent days")
plt.xlabel("Frequent days")

plt.subplot(323)
plt.hist(train['doy'], bins=BIN)
plt.title("Histogram of frequent doys")
plt.xlabel("Frequent doys")

plt.subplot(324)
plt.hist(train['wday'], bins=BIN)
plt.title("Histogram of frequent week days")
plt.xlabel("Frequent week days")

plt.subplot(325)
plt.hist(train['minute'], bins=BIN)
plt.title("Histogram of frequent minutes")
plt.xlabel("Frequent minutes")

plt.subplot(326)
plt.hist(train['second'], bins=BIN)
plt.title("Histogram of frequent seconds")
plt.xlabel("Frequent seconds")
plt.show()


# In[15]:


most_freq_hours_in_data    = [4, 5, 9, 10, 13, 14]
middle1_freq_hours_in_data = [16, 17, 22]
least_freq_hours_in_data   = [6, 11, 15]
train['in_hh'] = (   4 
                     - 3*train['hour'].isin(  most_freq_hours_in_data ) 
                     - 2*train['hour'].isin(  middle1_freq_hours_in_data ) 
                     - 1*train['hour'].isin( least_freq_hours_in_data ) ).astype('uint8')

gp    = train[['ip', 'day', 'in_hh', 'channel']].groupby(by=['ip', 'day', 'in_hh'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_day_hh'})
train = train.merge(gp, on=['ip','day','in_hh'], how='left')
train.drop(['in_hh'], axis=1, inplace=True)
train['nip_day_hh'] = train['nip_day_hh'].astype('uint32')
del gp
gc.collect()


# In[16]:


print("Train new time parameters")
print(train.dtypes)


# In[17]:


# Define all the groupby transformations
GROUPBY_AGGREGATIONS = [
    # Variance in day, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var', 'type': 'float32'},
    # Variance in day, for ip-app-device
    {'groupby': ['ip','app','device'], 'select': 'day', 'agg': 'var', 'type': 'float32'},
    # Variance in day, for ip-app-os
    {'groupby': ['ip','app','os'], 'select': 'day', 'agg': 'var', 'type': 'float32'},
    
    # Variance in hour, for ip-app-channel
    #{'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-app-device
    #{'groupby': ['ip','app','device'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    #{'groupby': ['ip','app','os'], 'select': 'hour', 'agg': 'var'},

    # Count, for ip-day
    #{'groupby': ['ip','day'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-day
    #{'groupby': ['ip','day'], 'select': 'device', 'agg': 'count'},
    # Count, for ip-day
    #{'groupby': ['ip','day'], 'select': 'os', 'agg': 'count'},
    
    # Count, for ip-hour
   # {'groupby': ['ip','hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-hour
    #{'groupby': ['ip','hour'], 'select': 'device', 'agg': 'count'},
    # Count, for ip-hour
    #{'groupby': ['ip','hour'], 'select': 'os', 'agg': 'count'},

    # Count, for ip-day-hour
    {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'},
    # Count, for ip-day-hour
    #{'groupby': ['ip','day','hour'], 'select': 'device', 'agg': 'count', 'type': 'uint32'},
    # Count, for ip-day-hour
   # {'groupby': ['ip','day','hour'], 'select': 'os', 'agg': 'count', 'type': 'uint32'},
    
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'},        
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'},
    
    # Mean hour, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean', 'type': 'float32', 'type': 'float32'}
]
# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    print(f"Grouping by {spec['groupby']}, and aggregating {spec['select']} with {spec['agg']}")
    
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), spec['agg'], spec['select'])
     # Perform the groupby
    gp = train[all_features].         groupby(spec['groupby'])[spec['select']].         agg(spec['agg']).         reset_index().         rename(index=str, columns={spec['select']: new_feature}).astype(spec['type'])
     # Merge back to X_train
    train = train.merge(gp, on=spec['groupby'], how='left')
del gp
gc.collect()
print("End")


# In[18]:


print(train.dtypes)


# In[19]:


train['app']           = train['app'].astype('uint16')
train['channel']       = train['channel'].astype('uint16')
train['device']        = train['device'].astype('uint16')
train['ip']            = train['ip'].astype('uint32')
train['os']            = train['os'].astype('uint16')


# In[20]:


train_X  = train[:len_train].drop(['click_id', 'is_attributed'], axis=1)
train_y  = train[:len_train]['is_attributed'].astype('uint8')
test_X   = train[len_train:].drop(['click_id', 'is_attributed'], axis=1)
test_id  = train[len_train:]['click_id'].astype('int')
del train


# In[21]:


print(train_X.dtypes)


# In[22]:


#path_train_X = path_out + 'train_X.csv'
#path_train_y = path_out + 'train_y.csv'
#print('Loading the pre training data...')
#train_X = pd.read_csv(path_train_X, header=0)
#train_y = pd.read_csv(path_train_y, header=0)
#print('End loading pre train data...')
predictors  = ['app','device','os', 'channel', 'hour', 'day', 'doy', 'wday','minute','second',
               'ip_app_channel_var_day',
               'ip_app_device_var_day',
               'ip_app_os_var_day',
               'ip_day_hour_count_channel',
               'ip_app_count_channel',
               'ip_app_os_count_channel',
               'ip_app_day_hour_count_channel',
               'ip_app_channel_mean_hour',
              'nip_day_hh']
categorical = ['app','device','os', 'channel', 'hour', 'day', 'doy', 'wday','minute','second']           


# In[23]:


metrics = 'auc'
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':metrics,
        'learning_rate': 0.05,
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 8,
        'verbose': 0,
        'scale_pos_weight':99.7, # because training data is extremely unbalanced 
        'metric':metrics
}
 
early_stopping_rounds = 100
num_boost_round       = 10000

print("Preparing validation datasets")
train_X, val_X = train_test_split(train_X, train_size=.95, shuffle=False )
train_y, val_y = train_test_split(train_y, train_size=.95, shuffle=False )
print("End preparing validation datasets")

xgtrain = lgb.Dataset(train_X[predictors].values, label=train_y,feature_name=predictors,
                       categorical_feature=categorical)
xgvalid = lgb.Dataset(val_X[predictors].values, label=val_y,feature_name=predictors,
                      categorical_feature=categorical)
evals_results = {}
model_lgb     = lgb.train(lgb_params,xgtrain,valid_sets=[xgtrain, xgvalid], 
                          valid_names=['train','valid'], 
                           evals_result=evals_results, 
                           num_boost_round=num_boost_round,
                           early_stopping_rounds=early_stopping_rounds,
                           verbose_eval=10, feval=None)   



# In[24]:


print("Features importance...")
gain = model_lgb.feature_importance('gain')
ft = pd.DataFrame({'feature':model_lgb.feature_name(), 
                   'split':model_lgb.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft.head(50))
ft.to_csv('importance_lightgbm.csv',index=True)
plt.figure()
ft = ft.sort_values('gain', ascending=True)
ft[['feature','gain']].head(50).plot(kind='barh', x='feature', y='gain', legend=False, figsize=(10, 10))
plt.gcf().savefig('features_importance.png')


# In[25]:


sub = pd.DataFrame()
sub['click_id'] = test_id
print("Sub dimension "    + str(sub.shape))
print("Test_X dimension " + str(test_X.shape))


# In[27]:


print("Predicting...")
sub['is_attributed'] = model_lgb.predict(test_X[predictors])  #
print("Writing...")
sub.to_csv('sub_Yatsenko_01.csv',index=False)
print("Done...")


# In[ ]:


#train[:len_train].drop(['click_time', 'click_id', 'is_attributed'], axis=1).to_csv('train_X.csv', index=False)
#print('End saving train_X')
#train[:len_train]['is_attributed'].astype('uint8').to_csv('train_y.csv', index=False)
#print('End saving train_y')
#train[len_train:].drop(['click_time', 'click_id', 'is_attributed'], axis=1).to_csv('test_X.csv', index=False)
#print('End saving test_X')
#train[len_train:]['click_id'].astype('int').to_csv('test_ids.csv', index=False)
#print('End saving test_ids')
#del train


