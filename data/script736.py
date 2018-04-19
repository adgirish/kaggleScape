
# coding: utf-8

# We will do analysis of some basic information about this dataset. It will help in building our model/features. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)


# In[ ]:


# Read train file
train_df = pd.read_csv("../input/train.csv")


# In[ ]:


train_df.head()


# In[ ]:


#Let's compute the number of missing values in each column
train_df.isnull().sum(axis=0).reset_index()


# In[ ]:


# Let's have a look at test dataset
test_df = pd.read_csv("../input/test.csv")
test_df.shape


# It can be observed that test data have fewer columns. It does not contain dropoff datetime and obviously the target variable (trip_duration) :). 

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.trip_duration.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('trip duration', fontsize=12)
plt.show()


# In[ ]:


# in train dataset some trip duration are very high (I consider them outliers and remove them before replotting it)
q = train_df.trip_duration.quantile(0.99)
train_df = train_df[train_df.trip_duration < q]
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.trip_duration.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('trip duration', fontsize=12)
plt.show()


# Now it looks better. We can perform some additional analysis on it

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train_df.trip_duration.values, bins=50, kde=True)
plt.xlabel('trip_duration', fontsize=12)
plt.show()


# It looks like a right skewed distribution. One can apply logarithm to make it normally distributed. 

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(np.log(train_df.trip_duration.values), bins=50, kde=True)
plt.xlabel('trip_duration', fontsize=12)
plt.show()


# Just another way to visualize in Maps..

# In[ ]:


from mpl_toolkits.basemap import Basemap
from matplotlib import cm
west, south, east, north = -74.05, 40.70, -74.00, 40.75

samples = train_df.sample(n=10000)
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)
m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,
            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')
x, y = m(samples['pickup_longitude'].values, samples['pickup_latitude'].values)
m.hexbin(x, y, gridsize=1000,
         bins='log', cmap=cm.YlOrRd_r);


# Lets fix some latitudes and longitudes ..

# But, what about the possible coordiantes that point to the water? Let's plot the southwest area of the city.

# In[ ]:


pickup_x = train_df.pickup_longitude.values
pickup_y = train_df.pickup_latitude.values
dropoff_x = train_df.dropoff_longitude.values
dropoff_y = train_df.dropoff_latitude.values

sns.set_style('white')

fig, ax = plt.subplots(figsize=(11, 12))

ax.scatter(pickup_x, pickup_y, s=5, color='blue', alpha=0.5)
ax.scatter(dropoff_x, dropoff_y, s=5, color='red', alpha=0.5)

ax.set_xlim([-74.05, -74.00])
ax.set_ylim([40.70, 40.75])

ax.set_title('coordinates')

sns.set_style('darkgrid')


# So we can see that there are lot of points which represent water. We can fix this by correcting latitudes and longitudes...

# In[ ]:


def fix_location(data):
    data.pickup_latitude = ((data.pickup_latitude >= 40.459518) & (data.pickup_latitude <= 41.175342))
    data.pickup_longitude = ((data.pickup_longitude >= -74.361107) & (data.pickup_longitude <= -71.903083))
    data.dropoff_latitude = ((data.dropoff_latitude >= 40.459518) & (data.dropoff_latitude <= 41.175342))
    data.dropoff_longitude = ((data.dropoff_longitude >= -74.361107) & (data.dropoff_longitude <= -71.903083))
    return(data)
train_df=fix_location(train_df)
test_df=fix_location(test_df)


# Let's compute the distance using the pick_up, drop langitude lattitude information and analyse it's relation with trip duration

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


train_df['radial_distance'] = haversine_np(train_df.pickup_longitude, train_df.pickup_latitude,
                                           train_df.dropoff_longitude, train_df.dropoff_latitude)
test_df['radial_distance'] = haversine_np(test_df.pickup_longitude, test_df.pickup_latitude,
                                           train_df.dropoff_longitude, train_df.dropoff_latitude)


# It takes large amount of time to get a best fitting plot between radial distance and trip duration, so I am taking first 10000 examples and ploting it

# In[ ]:


sns.regplot(train_df.trip_duration[0:10000], train_df.radial_distance[0:10000])


# In[ ]:


# Let's compute pickup hour for each ride
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['pickup_hour'] = train_df.pickup_datetime.dt.hour
train_df['day_week'] = train_df.pickup_datetime.dt.weekday
# Get pick up hour for test data as well
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])
test_df['pickup_hour'] = test_df.pickup_datetime.dt.hour
test_df['day_week'] = test_df.pickup_datetime.dt.weekday


# Now lets derive some more directional features and also speed which will add to our model...

# In[ ]:


#Lets compute some more features :

def engineer_features(data):

    data['Direction_NS'] = (data.pickup_latitude>data.dropoff_latitude)*1+1
    indices = data[(data.pickup_latitude == data.dropoff_latitude) & (data.pickup_latitude!=0)].index
    data.loc[indices,'Direction_NS'] = 0

    # create direction variable Direction_EW. 
    # This is 2 if taxi moving from east to west, 1 in the opposite direction and 0 otherwise
    data['Direction_EW'] = (data.pickup_longitude>data.dropoff_longitude)*1+1
    indices = data[(data.pickup_longitude == data.dropoff_longitude) & (data.pickup_longitude!=0)].index
    data.loc[indices,'Direction_EW'] = 0
    
    # create variable for Speed
    #print("deriving Speed. Make sure to check for possible NaNs and Inf vals...")
    #data['Speed_mph'] = data.radial_distance/(data.trip_duration/60)
    
    # replace all NaNs values and values >240mph by a values sampled from a random distribution of 
    # mean 12.9 and  standard deviation 6.8mph. These values were extracted from the distribution
    #indices_oi = data[(data.Speed_mph.isnull()) | (data.Speed_mph>240)].index
    #data.loc[indices_oi,'Speed_mph'] = np.abs(np.random.normal(loc=12.9,scale=6.8,size=len(indices_oi)))
    print("Feature engineering done! :-)")
    return(data)

train_df=engineer_features(train_df)


# In[ ]:


for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_y = train_df.trip_duration.values
train_X = train_df.drop(["id", "dropoff_datetime", "pickup_datetime", "trip_duration"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


# ### Visualize distribution of pick up hour

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="pickup_hour", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('pick up hour', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# The distribution shows the car demand with pick up hour time. After mid night less number's of trips are taken. Now let us see how the trip duration changes with respect to trip time.

# In[ ]:


grouped_df = train_df.groupby('pickup_hour')['trip_duration'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,8))
sns.pointplot(grouped_df.pickup_hour.values, grouped_df.trip_duration.values, alpha=0.8, color=color[3])
plt.ylabel('median trip duration', fontsize=12)
plt.xlabel('pick up hour', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


# Group by day
grouped_df = train_df.groupby('day_week')['trip_duration'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,8))
sns.pointplot(grouped_df.day_week.values, grouped_df.trip_duration.values, alpha=0.8, color=color[3])
plt.ylabel('trip duration median', fontsize=12)
plt.xlabel('week day', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


# Number of trips per day of week
plt.figure(figsize=(12,8))
sns.countplot(x="day_week", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Week day (0 - Monday)', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:


from bokeh.plotting import output_notebook, figure, show
from bokeh.models import HoverTool, BoxSelectTool

output_notebook()

TOOLS = [BoxSelectTool(), HoverTool()]

p = figure(plot_width=600, plot_height=400, title='A test scatter plot with hover labels', tools=TOOLS)

p.circle([1, 2, 3, 4, 5], [2, 5, 8, 2, 7], size=10)

show(p)

