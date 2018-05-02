
# coding: utf-8

# In this notebook I'll try the approach, which discovered in one tutorial about multivariate time series forecasting using LSTM.
# 

# In[1]:


import pandas as pd
import numpy as np
np.random.seed(10)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM


# ## **Data Aggregation**
# 
# 
# Features from **the1owl**'s kernel https://www.kaggle.com/the1owl/surprise-me

# In[2]:


data = {
    'tra': pd.read_csv('../input/air_visit_data.csv'),
    'as': pd.read_csv('../input/air_store_info.csv'),
    'hs': pd.read_csv('../input/hpg_store_info.csv'),
    'ar': pd.read_csv('../input/air_reserve.csv'),
    'hr': pd.read_csv('../input/hpg_reserve.csv'),
    'id': pd.read_csv('../input/store_id_relation.csv'),
    'tes': pd.read_csv('../input/sample_submission.csv'),
    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])


# In[3]:


for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date    
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    data[df] = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date'})


# In[4]:


data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date


# In[5]:


unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
tmp = data['tra'].groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])


# In[6]:


stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 
lbl = LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 


# In[7]:


train = pd.merge(data['tra'], stores, how='left', on=['air_store_id','dow']) 
test = pd.merge(data['tes'], stores, how='left', on=['air_store_id','dow'])

for df in ['ar','hr']:
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])
    
train = train.fillna(-1)
test = test.fillna(-1)


# In[8]:


def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5


# In[9]:


train.head()


# # **Part 1** 
# ## **Visitors as a feature to fit LSTM**
# 
# Functions from https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras

# ### *Normalize feature*

# In[10]:


train = train.sort_values('visit_date')
values = np.log1p(train['visitors'].values).reshape(-1,1)
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# ### *Split into train and test sets*

# In[11]:


train_size = int(len(scaled) * 0.7)
test_size = len(scaled) - train_size

V_train, V_test = scaled[0:train_size,:], scaled[train_size:len(scaled),:]
print(len(V_train), len(V_test))


# ### *Convert an array of values into a dataset matrix*

# In[12]:


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    print(len(dataY))
    return np.array(dataX), np.array(dataY)


# ### *Create dataset with look back*

# In[13]:


look_back = 1
trainX, trainY = create_dataset(V_train, look_back)
testX, testY = create_dataset(V_test, look_back)


# ### *Reshape X for model training*

# In[14]:


trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


#  ### *Train LSTM with 3 epochs*

# In[15]:


model = Sequential()
model.add(LSTM(4, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history = model.fit(trainX, trainY, epochs=3, batch_size=100,
                            validation_data=(testX, testY), verbose=1, shuffle=False) 


# ### *Make prediction and apply invert scaling*

# In[16]:


yhat = model.predict(testX)

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
testY_inverse = scaler.inverse_transform(testY.reshape(-1, 1))


# ### *RMSLE*

# In[17]:


rmsle = RMSLE(testY_inverse, yhat_inverse)
print('Test RMSLE: %.3f' % rmsle)


# # **Part 2**
# ## **Multivariate Forecast**
# 
# Functions from https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras

# ### *Using all features for model training*

# In[18]:


train = train.sort_values('visit_date')
target_train = np.log1p(train['visitors'].values)

col = [c for c in train if c not in ['id', 'air_store_id', 'visitors']]

train = train[col]
train.set_index('visit_date', inplace=True)

train.head()


# ### *Function to convert series to supervised learning*

# In[19]:


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# ### *Normalize features*

# In[20]:


train['visitors'] = target_train
values = train.values
values = values.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# ### *Frame as supervised learning*

# In[21]:


reframed = series_to_supervised(scaled, 1, 1)
reframed.head()


# ### *Drop unncessary columns*

# In[22]:


reframed.drop(reframed.columns[[i for i in range(17,33)]], axis=1, inplace=True)
reframed.head()


# ### *Split into train and test sets*

# In[23]:


values = reframed.values
n_train_days = int(len(values) * 0.7)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# Split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# Reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#  ### *Train LSTM with 3 epochs*

# In[24]:


multi_model = Sequential()
multi_model.add(LSTM(4, input_shape=(train_X.shape[1], train_X.shape[2])))
multi_model.add(Dense(1))
multi_model.compile(loss='mse', optimizer='adam')
multi_history = multi_model.fit(train_X, train_y, epochs=3,
                                batch_size=100, validation_data=(test_X, test_y),
                                verbose=1, shuffle=False)


# ### *Make prediction*

# In[25]:


yhat = multi_model.predict(test_X)


# ### *Apply invert scaling*

# In[26]:


test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# Invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# Invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]


# ### *RMSLE*

# In[27]:


rmsle = RMSLE(inv_y, inv_yhat)
print('Test RMSLE: %.3f' % rmsle)


# Slight improve, not enough, however, to beat benchmarks. I would add that the LSTM may not be suited for autoregression type problems (at least with such set of features, window and LSTM-configuration) and that maybe better off exploring an MLP with a large window. But i think it's good demo how to fit neural network to a multivariate time series forecasting problem. 
# Specifically:
# 
# * How to transform a raw dataset into something we can use for time series forecasting.
# * How to prepare data and fit an LSTM for a multivariate time series forecasting problem.
# * How to make a forecast and rescale the result back into the original units.
