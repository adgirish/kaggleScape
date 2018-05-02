
# coding: utf-8

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


# In[ ]:


df = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')
df.pickup_datetime=pd.to_datetime(df.pickup_datetime)
df.dropoff_datetime=pd.to_datetime(df.dropoff_datetime)
df['pu_hour'] = df.pickup_datetime.dt.hour
df['yday'] = df.pickup_datetime.dt.dayofyear
df['wday'] = df.pickup_datetime.dt.dayofweek


# In[ ]:


df.head()


# In[ ]:


wdf = pd.read_csv('../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016.csv')


# In[ ]:


wdf['date']=pd.to_datetime(wdf.date,format='%d-%m-%Y')
wdf['yday'] = wdf.date.dt.dayofyear


# In[ ]:


wdf.head()


# In[ ]:


falls = [ 0.01 if c=='T' else float(c) for c in wdf['snow fall']]
rain = [ 0.01 if c=='T' else float(c) for c in wdf['precipitation']]
wdf['snow fall']= falls
wdf['precipitation'] = rain


# In[ ]:


df = pd.merge(df,wdf,on='yday')
df.head()


# In[ ]:


df = df.drop(['date','maximum temerature','minimum temperature'],axis=1)
df.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.subplots(1,1,figsize=(17,15))
rain = wdf['precipitation']
sns.barplot(wdf['yday'], rain)


# In[ ]:


intensity = wdf['precipitation'].apply(lambda x:'L' if x < 0.098 
                           else 'M' if x>=0.098 and x<0.30 
                           else 'H' if x>=0.30 and x<2.0
                           else 'V')
wdf['precipitation'] = intensity
rain_count = wdf['precipitation'].value_counts().sort_values()
plt.subplots(1,1,figsize=(17,10))
sns.barplot(rain_count.index,rain_count.values)


# In[ ]:


rain_dummies = pd.get_dummies(wdf['precipitation'])
df = pd.concat([df, rain_dummies], axis=1)
df = df.drop(['precipitation'],axis=1)


# In[ ]:


df.head()


# ## Locations

# In[ ]:


day =1
df_day=df[((df.pickup_datetime<'2016-02-'+str(day+1))&
           (df.pickup_datetime>='2016-02-'+str(day)))]


# In[ ]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))
plt.ylim(40.6, 40.9)
plt.xlim(-74.1,-73.7)
ax.scatter(df['pickup_longitude'],df['pickup_latitude'], s=0.0002, alpha=1)


# ## Distances

# As described in this great kernel https://www.kaggle.com/skhemka/exploratory-data-analysis there are outliers in the trip distances so we remove them.

# In[ ]:


#plt.figure(figsize=(8,6))
f,axarr = plt.subplots(ncols=2,nrows=1,figsize=(12,6))
axarr[0].scatter(range(df.shape[0]), np.sort(df.trip_duration.values))
q = df.trip_duration.quantile(0.99)
df = df[df.trip_duration < q]
axarr[1].scatter(range(df.shape[0]), np.sort(df.trip_duration.values))

plt.show()


# and compute the distance

# In[ ]:


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


# In[ ]:


df['distance'] = haversine_np(df.pickup_longitude, df.pickup_latitude,
                                           df.dropoff_longitude, df.dropoff_latitude)


# In[ ]:


import seaborn as sns
#sns.set(style="ticks")
sel = df[['distance','passenger_count']]
sns.barplot(x='passenger_count',y='distance',data=sel)
#sns.despine(offset=10, trim=True)


# In[ ]:


import seaborn as sns
#sns.set(style="ticks")
sel = df[['distance','wday']]
sns.barplot(x='wday',y='distance',data=sel)
#sns.despine(offset=10, trim=True)


# ## Attempting Regression

# I am going to start with a linear regression model and I will make several attempts with base and engineered features.

# In[ ]:


features = df[['wday','yday','pu_hour','passenger_count','pickup_latitude','pickup_longitude','vendor_id']]
target = df[['trip_duration']]


# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

reg = linear_model.LinearRegression()
cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=0)
cross_val_score(reg, features, target, cv=cv)
#reg.fit (features, target)


# In[ ]:


reg = linear_model.Ridge (alpha = .5)
cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=0)
cross_val_score(reg, features, target, cv=cv)


# In[ ]:


reg.fit(features,target)


# In[ ]:


tdf = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
tdf.pickup_datetime=pd.to_datetime(tdf.pickup_datetime)
#tdf.dropoff_datetime=pd.to_datetime(tdf.dropoff_datetime)
tdf['pu_hour'] = tdf.pickup_datetime.dt.hour
tdf['yday'] = tdf.pickup_datetime.dt.dayofyear
tdf['wday'] = tdf.pickup_datetime.dt.dayofweek


# In[ ]:


tfeatures = tdf[['wday','yday','pu_hour','passenger_count','pickup_latitude','pickup_longitude','vendor_id']]


# In[ ]:


pred = reg.predict(tfeatures)


# In[ ]:


tdf['trip_duration']=pred.astype(int)
out = tdf[['id','trip_duration']]


# In[ ]:


out['trip_duration'].isnull().values.any()


# In[ ]:


out.to_csv('pred_linear_1.csv',index=False)


# This model performed very bad :( scoring only 0.808. Let's try something more sophisticated. We can identify clusters of pick up locations and use them to train the model.

# ##Creating Clusters

# In[ ]:


from sklearn.cluster import KMeans
import numpy as np
import pickle

try:
    kmeans = pickle.load(open("source_kmeans.pickle", "rb"))
except:
    kmeans = KMeans(n_clusters=20, random_state=0).fit(df[['pickup_longitude','pickup_latitude']])
    pickle.dump(kmeans, open('source_kmeans.pickle', 'wb'))

try:
    destkmeans = pickle.load(open("dest_kmeans.pickle", "rb"))
except:
    destkmeans = KMeans(n_clusters=20, random_state=0).fit(df[['dropoff_longitude','dropoff_latitude']])
    pickle.dump(destkmeans, open('dest_kmeans.pickle', 'wb'))


# In[ ]:


cx = [c[0] for c in kmeans.cluster_centers_]
cy = [c[1] for c in kmeans.cluster_centers_]


# In[ ]:


fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))
plt.ylim(40.6, 40.9)
plt.xlim(-74.1,-73.7)

df['cluster'] = kmeans.predict(df[['pickup_longitude','pickup_latitude']])
df['dest_cluster'] = destkmeans.predict(df[['dropoff_longitude','dropoff_latitude']])
cm = plt.get_cmap('gist_rainbow')

colors = [cm(2.*i/15) for i in range(20)]
colored = [colors[k] for k in df['cluster']]

#plt.figure(figsize = (10,10))
ax.scatter(df.pickup_longitude,df.pickup_latitude,color=colored,s=0.0002,alpha=1)
ax.scatter(cx,cy,color='Black',s=50,alpha=1)
plt.title('Taxi Pickup Clusters')
plt.show()
#plt.ylim(40.6, 40.9)

#ax.scatter(sdf['pickup_longitude'],sdf['pickup_latitude'], s=0.1, alpha=1)
#ax.scatter(cx,cy,s=70,color='Red')


# In[ ]:


fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(12,10))
plt.ylim(40.6, 40.9)
plt.xlim(-74.1,-73.7)

cx = [c[0] for c in destkmeans.cluster_centers_]
cy = [c[1] for c in destkmeans.cluster_centers_]

colors = [cm(2.*i/15) for i in range(20)]
colored = [colors[k] for k in df['dest_cluster']]

ax.scatter(df.dropoff_longitude,df.dropoff_latitude,color=colored,s=0.0002,alpha=1)
ax.scatter(cx,cy,color='Black',s=50,alpha=1)
plt.title('Taxi Dropoff Clusters')
plt.show()


# In[ ]:


df.head()
features = df[['wday','yday','pu_hour','passenger_count','cluster','vendor_id','dest_cluster']]
target = df[['trip_duration']]


# In[ ]:


from sklearn import linear_model
import xgboost
reg = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

reg.fit(features,target)


# In[ ]:


tdf = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')
tdf['cluster'] = kmeans.predict(tdf[['pickup_longitude','pickup_latitude']])
tdf['dest_cluster'] = destkmeans.predict(tdf[['dropoff_longitude','dropoff_latitude']])
tdf.pickup_datetime=pd.to_datetime(tdf.pickup_datetime)
#tdf.dropoff_datetime=pd.to_datetime(tdf.dropoff_datetime)
tdf['pu_hour'] = tdf.pickup_datetime.dt.hour
tdf['yday'] = tdf.pickup_datetime.dt.dayofyear
tdf['wday'] = tdf.pickup_datetime.dt.dayofweek
tfeatures = tdf[['wday','yday','pu_hour','passenger_count','cluster','vendor_id','dest_cluster']]
pred = reg.predict(tfeatures)


# In[ ]:


tdf['trip_duration']=pred.astype(int)
out = tdf[['id','trip_duration']]
out['trip_duration'].isnull().values.any()
out.to_csv('pred_linear_2_clusters.csv',index=False)


# I have added both the pickup and dropoff clusters which is very unfair with respect to the first model since the dropoff information is obviously correlated with the duration.
# This model scored 0.621. 
# Now I am going to us weather conditions

# ##Weather Doesn't Matter

# In[ ]:


features = df[['wday','yday','pu_hour','passenger_count','cluster','vendor_id','dest_cluster','H','L','M','V','snow fall']]
target = df[['trip_duration']]


# In[ ]:


reg = xgboost.XGBRegressor(n_estimators=200, min_child_weight=150, gamma=0, subsample=0.75,
                           colsample_bytree=0.4, max_depth=10)

reg.fit(features,target)


# In[ ]:


tdf = pd.merge(tdf,wdf,on='yday')
tdf.head()


# In[ ]:


rain_dummies = pd.get_dummies(tdf['precipitation'])
tdf = pd.concat([tdf, rain_dummies], axis=1)
tdf = tdf.drop(['precipitation'],axis=1)
tfeatures = tdf[['wday','yday','pu_hour','passenger_count','cluster','vendor_id','dest_cluster','H','L','M','V','snow fall']]

pred = reg.predict(tfeatures)


# In[ ]:


pred = reg.predict(tfeatures)


# In[ ]:


tdf['trip_duration']=pred.astype(int)
out = tdf[['id','trip_duration']]
out['trip_duration'].isnull().values.any()
out.to_csv('pred_3_clusters_weather.csv',index=False)

