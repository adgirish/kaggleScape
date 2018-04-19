
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Read Data

# In[ ]:


df_all = pd.concat((pd.read_csv('../input/train.csv'), pd.read_csv('../input/train.csv')))


# In[ ]:


df_all['pickup_datetime'] = df_all['pickup_datetime'].apply(pd.Timestamp)
df_all['dropoff_datetime'] = df_all['dropoff_datetime'].apply(pd.Timestamp)


# In[ ]:


df_all['trip_duration_log'] = df_all['trip_duration'].apply(np.log)


# # Feature Extraction

# ## PCA
# lets use PCA just to rotate NYC to align with axes (lets help those trees...)

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


X = np.vstack((df_all[['pickup_latitude', 'pickup_longitude']], 
               df_all[['dropoff_latitude', 'dropoff_longitude']]))

# Remove abnormal locations
min_lat, min_lng = X.mean(axis=0) - X.std(axis=0)
max_lat, max_lng = X.mean(axis=0) + X.std(axis=0)
X = X[(X[:,0] > min_lat) & (X[:,0] < max_lat) & (X[:,1] > min_lng) & (X[:,1] < max_lng)]


# In[ ]:


pca = PCA().fit(X)
X_pca = pca.transform(X)


# In[ ]:


_, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,9))

sample_ind = np.random.permutation(len(X))[:10000]

ax1.scatter(X[sample_ind,0], X[sample_ind,1], s=1, lw=0)
ax1.set_title('Original')

ax2.scatter(X_pca[sample_ind,0], X_pca[sample_ind,1], s=1, lw=0)
ax2.set_title('Rotated')


# In[ ]:


df_all['pickup_pca0'] = pca.transform(df_all[['pickup_latitude', 'pickup_longitude']])[:,0]
df_all['pickup_pca1'] = pca.transform(df_all[['pickup_latitude', 'pickup_longitude']])[:,1]

df_all['dropoff_pca0'] = pca.transform(df_all[['dropoff_latitude', 'dropoff_longitude']])[:,0]
df_all['dropoff_pca1'] = pca.transform(df_all[['dropoff_latitude', 'dropoff_longitude']])[:,1]


# ## Distances

# In[ ]:


df_all['pca_manhattan'] =     (df_all['dropoff_pca0'] - df_all['pickup_pca0']).abs() +     (df_all['dropoff_pca1'] - df_all['pickup_pca1']).abs()


# In[ ]:


def arrays_haversine(lats1, lngs1, lats2, lngs2, R=6371):
    lats1_rads = np.radians(lats1)
    lats2_rads = np.radians(lats2)
    lats_delta_rads = np.radians(lats2 - lats1)
    lngs_delta_rads = np.radians(lngs2 - lngs1)
    
    a = np.sin(lats_delta_rads / 2)**2 + np.cos(lats1) * np.cos(lats2) * np.sin(lngs_delta_rads / 2)**2
    c = 2 * np.arcsin(a**0.5)
    
    return R * c


# In[ ]:


df_all['haversine'] = arrays_haversine(
    df_all['pickup_latitude'], df_all['pickup_longitude'], 
    df_all['dropoff_latitude'], df_all['dropoff_longitude'])


# In[ ]:


def arrays_bearing(lats1, lngs1, lats2, lngs2, R=6371):
    lats1_rads = np.radians(lats1)
    lats2_rads = np.radians(lats2)
    lngs1_rads = np.radians(lngs1)
    lngs2_rads = np.radians(lngs2)
    lngs_delta_rads = np.radians(lngs2 - lngs1)
    
    y = np.sin(lngs_delta_rads) * np.cos(lats2_rads)
    x = np.cos(lats1_rads) * np.sin(lats2_rads) - np.sin(lats1_rads) * np.cos(lats2_rads) * np.cos(lngs_delta_rads)
    
    return np.degrees(np.arctan2(y, x))


# In[ ]:


df_all['bearing'] = arrays_bearing(
    df_all['pickup_latitude'], df_all['pickup_longitude'], 
    df_all['dropoff_latitude'], df_all['dropoff_longitude'])


# ## Date-Time

# In[ ]:


df_all['pickup_time_delta'] = (df_all['pickup_datetime'] -                                df_all['pickup_datetime'].min()).dt.total_seconds()


# In[ ]:


df_all['month'] = df_all['pickup_datetime'].dt.month
df_all['weekofyear'] = df_all['pickup_datetime'].dt.weekofyear
df_all['weekday'] = df_all['pickup_datetime'].dt.weekday
df_all['hour'] = df_all['pickup_datetime'].dt.hour


# In[ ]:


df_all['week_delta'] =     df_all['pickup_datetime'].dt.weekday +     ((df_all['pickup_datetime'].dt.hour + (df_all['pickup_datetime'].dt.minute / 60.0)) / 24.0)


# In[ ]:


# Make time features cyclic
df_all['week_delta_sin'] = np.sin((df_all['week_delta'] / 7) * np.pi)**2
df_all['hour_sin'] = np.sin((df_all['hour'] / 24) * np.pi)**2


# ## Traffic

# In[ ]:


# Count trips over 60min
df_counts = df_all.set_index('pickup_datetime')[['id']].sort_index()
df_counts['count_60min'] = df_counts.isnull().rolling('60min').count()['id']
df_all = df_all.merge(df_counts, on='id', how='left')


# ## Clusters Info

# cluster trips in order to aggregate information about them

# In[ ]:


from sklearn.cluster import MiniBatchKMeans


# In[ ]:


kmeans = MiniBatchKMeans(n_clusters=8**2, batch_size=32**3).fit(X)


# In[ ]:


sample_ind = np.random.permutation(len(X))[:10000]
plt.scatter(X[sample_ind,0], X[sample_ind,1], s=1, lw=0, 
            c=kmeans.predict(X[sample_ind]), cmap='tab20')


# In[ ]:


df_all['pickup_cluster'] = kmeans.predict(df_all[['pickup_latitude', 'pickup_longitude']])
df_all['dropoff_cluster'] = kmeans.predict(df_all[['dropoff_latitude', 'dropoff_longitude']])


# In[ ]:


# Count how many trips are going to each cluster over time
group_freq = '60min'

df_dropoff_counts = df_all     .set_index('pickup_datetime')     .groupby([pd.TimeGrouper(group_freq), 'dropoff_cluster'])     .agg({'id': 'count'})     .reset_index().set_index('pickup_datetime')     .groupby('dropoff_cluster').rolling('240min').mean()     .drop('dropoff_cluster', axis=1)     .reset_index().set_index('pickup_datetime').shift(freq='-120min').reset_index()     .rename(columns={'pickup_datetime': 'pickup_datetime_group', 'id': 'dropoff_cluster_count'})
    
df_all['pickup_datetime_group'] = df_all['pickup_datetime'].dt.round(group_freq)

df_all['dropoff_cluster_count'] =     df_all[['pickup_datetime_group', 'dropoff_cluster']].merge(df_dropoff_counts, 
        on=['pickup_datetime_group', 'dropoff_cluster'], how='left')['dropoff_cluster_count'].fillna(0)


# ## Features Correlations

# In[ ]:


plt.figure(figsize=(16,9))
sns.heatmap(df_all.corr()[['trip_duration_log']].sort_values('trip_duration_log'), annot=True)


# In[ ]:


features = [
    'vendor_id', 'passenger_count',
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude',
    'pickup_pca0', 'pickup_pca1', 'dropoff_pca0', 'dropoff_pca1',
    'haversine', 'bearing', 'pca_manhattan',
    'pickup_time_delta', 'month', 'weekofyear', 'weekday', 'hour', 
    'week_delta', 'week_delta_sin', 'hour_sin',
    'count_60min', 'dropoff_cluster_count'
]


# # Train Models

# In[ ]:


from xgboost import XGBRegressor
from sklearn.model_selection import KFold


# In[ ]:


X_train = df_all[df_all['trip_duration'].notnull()][features].values
y_train = df_all[df_all['trip_duration'].notnull()]['trip_duration_log'].values

X_test = df_all[df_all['trip_duration'].isnull()][features].values


# Do 2 splits CV and predict the average of the two for the test data

# In[ ]:


xgb = XGBRegressor(n_estimators=1000, max_depth=12, min_child_weight=150, 
                   subsample=0.7, colsample_bytree=0.3)

y_test = np.zeros(len(X_test))

for i, (train_ind, val_ind) in enumerate(KFold(n_splits=2, shuffle=True, 
                                               random_state=1989).split(X_train)):
    print('----------------------')
    print('Training model #%d' % i)
    print('----------------------')
    
    xgb.fit(X_train[train_ind], y_train[train_ind],
            eval_set=[(X_train[val_ind], y_train[val_ind])],
            early_stopping_rounds=10, verbose=25)
    
    y_test += xgb.predict(X_test, ntree_limit=xgb.best_ntree_limit)
    
y_test /= 2


# # Submission

# In[ ]:


df_sub = pd.DataFrame({
    'id': df_all[df_all['trip_duration'].isnull()]['id'].values,
    'trip_duration': np.exp(y_test)}).set_index('id')


# In[ ]:


df_sub.to_csv('subs/sub001.csv')

