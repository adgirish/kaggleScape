
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, LSTM
from subprocess import check_output


# In[ ]:


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
data['hr'].drop('hpg_store_id',  axis=1, inplace=True)
data['ar'] = data['ar'].append(data['hr'])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes'].drop('id', axis=1, inplace=True)
print ('Data loaded')


# In[ ]:


# Create single data set with all relevant base data:
data['tra']['visit_datetime'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['visit_date']     = data['tra']['visit_datetime'].dt.date
data['ar']['res_visit_datetime'] = pd.to_datetime(data['ar']['visit_datetime'])
data['ar']['reserve_datetime']   = pd.to_datetime(data['ar']['reserve_datetime'])
data['ar']['visit_date']         = data['ar']['res_visit_datetime'].dt.date
data['ar']['reserve_diff'] = data['ar'].apply(lambda r: (r['res_visit_datetime']
                                                         - r['reserve_datetime']).days, 
                                        axis=1)
data['ar'].drop('visit_datetime',  axis=1, inplace=True)
data['ar'].drop('reserve_datetime',  axis=1, inplace=True)
data['ar'].drop('res_visit_datetime',  axis=1, inplace=True)
avg_reserv = data['ar'].groupby(['air_store_id','visit_date'], 
                                as_index=False).mean().reset_index()
data['ar'] = data['ar'].groupby(['air_store_id','visit_date'], 
                                as_index=False).sum().reset_index()
data['ar'] = data['ar'].drop(['reserve_diff'],axis=1)
data['ar'] = data['ar'].drop(['index'],axis=1)
data['ar']['reserve_diff'] = avg_reserv['reserve_diff']                            
    
data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date

data['tes']['visit_datetime'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['visit_date'] = data['tes']['visit_datetime'].dt.date

prep_df = pd.merge(data['tra'], data['ar'],  how='left', on=['air_store_id', 'visit_date'])
prep_df = pd.merge(prep_df,     data['as'],  how='inner', on='air_store_id')
prep_df = pd.merge(prep_df,     data['hol'], how='left',  on='visit_date')

predict_data = pd.merge(data['tes'],  data['ar'],   how='left', on=['air_store_id', 'visit_date'])
predict_data = pd.merge(predict_data, data['as'],   how='inner', on='air_store_id')
predict_data = pd.merge(predict_data, data['hol'],  how='left', on='visit_date')

#print(len(prep_df[prep_df.air_store_id == "air_35512c42db0868da"]))
#print(len(data['tra'][data['tra'].air_store_id == "air_35512c42db0868da"]))


# In[ ]:


# Encode fields:
prep_df['month'] = prep_df['visit_datetime'].dt.month
prep_df['day']   = prep_df['visit_datetime'].dt.day
prep_df.drop('visit_datetime',      axis=1, inplace=True)   
predict_data['month'] = predict_data['visit_datetime'].dt.month
predict_data['day']   = predict_data['visit_datetime'].dt.day
predict_data.drop('visit_datetime', axis=1, inplace=True)

# Encode labels of categorical columns:
cat_features = [col for col in ['air_genre_name', 'air_area_name', 'day_of_week']]
for column in cat_features:
    temp_prep = pd.get_dummies(pd.Series(prep_df[column]))
    prep_df = pd.concat([prep_df,temp_prep],axis=1)
    prep_df = prep_df.drop([column],axis=1)
    temp_predict = pd.get_dummies(pd.Series(predict_data[column]))
    predict_data = pd.concat([predict_data,temp_predict],axis=1)
    predict_data = predict_data.drop([column],axis=1)
    for missing_col in temp_prep:     # Make sure the columns of train and test are identical
        if missing_col not in predict_data.columns:
            predict_data[missing_col] = 0
    
prep_df['visitors'] = np.log1p(prep_df['visitors'])
prep_df.fillna(0, inplace=True)
predict_data.fillna(0, inplace=True)
print('Done')


# In[ ]:


air_ids = [air for air in prep_df['air_store_id'].unique()]
mult_series = dict()
scaler = MinMaxScaler(feature_range=(0, 1))

store_key = prep_df[['air_store_id', 'visit_date']]
store_key_predict = predict_data[['air_store_id', 'visit_date']]
prep_df.drop(['air_store_id', 'visit_date'], axis=1, inplace=True)  
predict_data.drop(['air_store_id', 'visit_date'], axis=1, inplace=True) 
cols = prep_df.columns
cols_predict = predict_data.columns
scaler.fit(prep_df)
scaled_prep_df      = pd.DataFrame(scaler.transform(prep_df), columns=cols)
scaled_predict_data = pd.DataFrame(scaler.transform(predict_data), columns=cols_predict)
scaled_prep_df['air_store_id'] = store_key['air_store_id']
scaled_prep_df['visit_date']   = store_key['visit_date']
scaled_predict_data['air_store_id'] = store_key_predict['air_store_id']
scaled_predict_data['visit_date']   = store_key_predict['visit_date']
scaled_predict_data['visitors'] = 0

for air_id in air_ids:
    tmp = pd.DataFrame(scaled_prep_df[scaled_prep_df['air_store_id'] == air_id]).sort_values('visit_date')
    tmp.drop('air_store_id', axis=1, inplace=True)  
    tmp.set_index('visit_date', inplace=True)
    mult_series[str(air_id)] = tmp.astype('float32')

mult_series['air_ee3a01f0c71a769f'].head(10)  # Print data for sample restaurant
#list(mult_series.keys())
# Target:  y = prep_df['visitors'].values


# In[ ]:


# From https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
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

# Submissions are evaluated using RMSLE:
def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5


# In[ ]:


# Convert data series for supervised learning:
tmp = pd.DataFrame(series_to_supervised(mult_series['air_ee3a01f0c71a769f'], 1, 1))
tmp.drop(tmp.columns[[i for i in range(133,264)]], axis=1, inplace=True)
super_data = tmp
for air_id in air_ids:
    tmp = series_to_supervised(mult_series[str(air_id)], 1, 1)
    # Drop columns that should not be predicted (column #103 is number of visitors:
    tmp.drop(tmp.columns[[i for i in range(133,264)]], axis=1, inplace=True)
    super_data = super_data.append(tmp)
super_data.head(10)


# In[ ]:


# Prepare LSTM training, split up records into training and test data:
train_size = int(len(super_data) * 0.7)
test_size = len(super_data) - train_size

train = super_data[:train_size].values
test  = super_data[train_size:].values

# Split into input and outputs
train_X, train_y = train[:,:-1], train[:,-1]
test_X, test_y = test[:, :-1], test[:, -1]

# LSTM requires 3D data sets: [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[ ]:


# Train model:
multi_model = Sequential()
btc_size = 50
#multi_model.add(LSTM(4, input_shape=(train_X.shape[1], train_X.shape[2])))
multi_model.add(LSTM(4, batch_input_shape=(btc_size, train_X.shape[1], train_X.shape[2]), 
                     stateful=True))
multi_model.add(Dense(1))
multi_model.compile(loss='mse', optimizer='adam')
for i in range(int(train_X.shape[0] / btc_size)):
    this_X = train_X[(i * btc_size):((i + 1) * btc_size)][:][:]
    this_y = train_y[(i * btc_size):((i + 1) * btc_size)]
    multi_history = multi_model.fit(this_X, this_y, epochs=10, 
                                batch_size=btc_size, 
                                verbose=0, shuffle=False)
    multi_model.reset_states()
  


# In[ ]:


# Make predictions:
y_pred = [test_X.shape[0]]
for i in range(int(test_X.shape[0] / btc_size)):
    this_X = test_X[(i * btc_size):((i + 1) * btc_size)][:][:]
    this_pred = multi_model.predict(this_X, batch_size=btc_size)    
    y_pred[(i * btc_size):((i + 1) * btc_size)] = this_pred

test_X_nn = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# Invert scaling for forecast
inv_y_pred = np.concatenate((y_pred, test_X_nn[:, 1:]), axis=1)
inv_y_pred = scaler.inverse_transform(inv_y_pred)
inv_y_pred = inv_y_pred[:,0]
# Invert scaling for actual
test_y_nn = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y_nn, test_X_nn[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

print(inv_y_pred[:10])
print(inv_y[:10])
rmsle = RMSLE(inv_y, inv_y_pred)
print('Test RMSLE: %.3f' % rmsle)

