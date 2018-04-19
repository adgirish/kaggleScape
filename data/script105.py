
# coding: utf-8

# <h1>An MVP Aproach</h1>
# =========
# <br>
# <h2> 1. Introduction</h2>
# ---------------------------------
# <p>As a (admittedly lazy) Data Scientist I love accessing information in the easiest way possible. Google is my best friend, for example. With the exception of a few things, there is almost nothing you can't find with a simple search.</p>
# 
# <p>This kernel is written and developed using IPython Notebook and XGBoost, with the aim of being ready-to-submit. By that I mean any user can run each code block, submit the output, and achieve a comfortable top-half score. Breaking into the top 10% of the submissions would require quite a bit more effort and a more sophisticated approach. In the final section I include some comments and suggestions of some alterations/variations one might try to improve the score. Those suggestions, if carried out correctly, do result in a strong score on the leaderboard. </p>
# 
# <p>This then serves as an introductory tutorial to approaching this particular problem (rather than giving the best solution outright) leaving enough room for creativity and innovation in the solution space to allow any user to better the end result.<br>
# In short, the whole project is about LEARNING, sharing some ideas, and ultimately improving the degree of Data Scientists out there. </p>
# <p>Enjoy! </p><br>
# 
# <h3>1.1 The Problem</h3> <br>
# <p>New York is riddled with one-ways, small side streets, and an almost incalculable amount of pedestrians at any given point in time. Not to mention the amount of cars/motorcycles/bicycles clogging up the roads. Combine this with a mad rush to get from point A to point B, and you'll find yourself late for whatever you need to be on time for.</p>
# 
# <p>The solution to getting from A to B when living in a city like New York (without losing your mind) is easy: take a taxi/Uber/Lyft/etc. You don't need to stress about the traffic or pedestrians and you have a moment to do something else, like catch up on emails. Although this sounds simple enough, it doesn't mean you'll get to your destination in time. So you need to have your driver take the shortest trip possible. By shortest, we're talking time. If a route A is X kilometers *longer*, but gets you there Y minutes *faster* than route B would, rather take that one.</p>
# 
# <p>To know which route is the best one to take, we need to be able to predict how long the trip will last when taking a specific route. Therefore, *the goal of this playground competition is to predict the the duration of each trip in the test data set, given start and end coordinates.*</p><br>
# 
# <h3>1.2 The Libraries & Functions</h3><br>
# <p>
# Using Python 3.6.1, import the following libraries. Note the use of `%matplotlib inline`, allowing the display of graphs inline in iPython Notebook.</p>
# 
# <h3>Documentation</h3>
# [Scikit-Learn](http://scikit-learn.org/stable/documentation.html "Scikit-Learn")<br>
# [Pandas](http://pandas.pydata.org/pandas-docs/stable/ "Pandas")<br>
# [Numpy](https://docs.scipy.org/doc/ "Numpy")<br>
# [XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html "XGBoost")<br>
# [Seaborn](https://seaborn.pydata.org/index.html "Seaborn")<br>
# 
# <p>I used Scikit-Learn (or sklearn) for a few of the machine learning operations that was carried out. Pandas is used for data manipulation. Numpy is the fundamental package for scientific computation in Python. XGBoost is the classification algorithm used to make the final predictions. Seaborn is a nice tool for data visualisation built on top of matplotlib. The import code is as follows:</p>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge,BayesianRidge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from math import radians, cos, sin, asin, sqrt
import seaborn as sns
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]


# <h2>1.3 Loading the Data</h2><br>
# <p>Load the data using the Pandas `read_csv` function:</p>

# In[ ]:


train = pd.read_csv('../input/new-york-city-taxi-with-osrm/train.csv')
test = pd.read_csv('../input/new-york-city-taxi-with-osrm/test.csv')


# <h2>1.4 Initial Data Exploration</h2><br>
# 
# <h3>1.4.1 File structure</h3><br>
# <p>Let's start off by exploring the files we just imported, using Pandas `head` and `describe` functions:</p>

# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
train.head()


# <p>Note the use of `pd.set_option('display.float_format', lambda x: '%.3f' % x)` to set the display options. More on that [here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.set_option.html "Pandas set_option").</p>
# 
# <p>You can also use `train.tail()` to display the last 5 entries in the two files. From the output we can get an idea of the way the data is structured. Of course, this isn't strictly necessary, since it is possible to get the same view by navigating to the *Input Files* at the top and clicking on any one of the CSV files used in this analysis.</p>
# <br>
# <h3>1.4.2 Data Summary</h3><br>
# <p>Getting a statistical summary of the data is also quite easy. This is where the `describe` function comes in:</p>

# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
train.describe()


# <p>Let's take a moment to think about what we're seeing here. What stands out? Initially nothing, until we get to the `trip_duration`. A minimum of 1 second, and a maximum of 3 526 282 seconds (roughly 980 hours). No way someone took a trip that long, the bill would be astronomical. And a 1 second trip won't get you anywhere. So clearly there are some outliers we need to deal with.</p>
# 
# <p>As a final way of looking at the data initially, consider the use of Pandas' `info` function. The advantage of using this particular function is getting an almost combined view of `describe` and `head`/`tail`, together with `shape` (which describes the number of rows and columns in the particular file). More on the use and output of `info` [here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.info.html "Pandas info").</p>

# In[ ]:


train.info()


# <p>So now that we have an idea of what the data looks like and the type of variables we're working with, what's next? This is where the fun really begins.</p>

# <h2>2. Data Preparation</h2>
# ---------------------
# <p>My initial thoughts on this is that from the pickup_datetime field we should be extracting hour of day and day of week and day of month.  This seems logical since peak hour traffic and weekend vs non-weekend days could have a mayor effect on travel times. Similar to this there could be seasonality in the travel times to be observed between the different months of the year. Think, for example, of the effect New York's winter months might have on travel time. If the roads are wet/icy you're less likely to drive very fast, so regardless of which route you're taking, you'll take a bit longer to reach your destination.</p>
# 
# <p>An interesting variable and its effects to potentially explore is the `passenger_count`. One might argue that an increased number of passengers could result in a few scenarios. It could, for example, result in increased stops along the way, which ultimately extends the time from A to B (note how we're not given the number of passengers at the start and end of the trip - my thinking is that if we were to know the start and end number of passengers in a trip, the data between drops would be split into separate entries, with the last entry in a collection of rows ending with 0 passengers). Also, from a purely physical point of view, the higher the number of passengers, the heavier the vehicle, the slower the vehicle might move. Although I'm fairly certain the effect of this is neglible. </p>
# 
# <p>Same with `vendor` and `store_and_fwd_flag`. It is possible that looking into it, we find that there's a difference between vendor 1 and vendor 2, and that either one of the two might be corrupting the "shortest route information" because the vendor's employees are less efficient in finding the best routes through New York. This, however, I also find highly unlikely, but it is an alternative to explore (at least look into it and rule it out difinitively). As for `store_and_fwd_flag` - not having a connection to the server for a particular route to be indicative of a few things. For example, If upon inspection it is discovered that there is a strong correlation between slow trip times and server disconnects, it could be used as another feature in the training model to predict what time a particular route could take.</p>
# 
# <p>Getting down to it however, the `_lattitude` and `_longitute` variables  is where I think the most value lies. The options here are to cluster them into "neighborhoods" as well as find the distance and directions between coordinates.</p>

# <h3>2.1 Trip Duration Clean-up</h3><br>
# 
# <p>As we noted earlier there are some outliers associated with the `trip_duration` variable, specifically a 980 hour maximum trip duration and a minimum of 1 second trip duration. I've decided to exclude data that lies outside 2 standard deviations from the mean. It might be worthwhile looking into what effect excluding up to 4 standard deviations would have on the end-results.</p>

# In[ ]:


m = np.mean(train['trip_duration'])
s = np.std(train['trip_duration'])
train = train[train['trip_duration'] <= m + 2*s]
train = train[train['trip_duration'] >= m - 2*s]


# [](http://)<h3>2.2 Latitude and Longitude Clean-up</h3><br>
# <p>Looking into it, the borders of NY City, in coordinates comes out to be:</p>
# 
# city_long_border = (-74.03, -73.75)<br>
# city_lat_border = (40.63, 40.85)
# <br>
# 
# <p>Comparing this to our `train.describe()` output we see that there are some coordinate points (pick ups/drop offs) that fall outside these borders. So let's limit our area of investigation to within the NY City borders.</p>

# In[ ]:


train = train[train['pickup_longitude'] <= -73.75]
train = train[train['pickup_longitude'] >= -74.03]
train = train[train['pickup_latitude'] <= 40.85]
train = train[train['pickup_latitude'] >= 40.63]
train = train[train['dropoff_longitude'] <= -73.75]
train = train[train['dropoff_longitude'] >= -74.03]
train = train[train['dropoff_latitude'] <= 40.85]
train = train[train['dropoff_latitude'] >= 40.63]


# [](http://)<h3>2.3 Date Clean-up</h3><br>
# <p>As a final step in preparing our data we need to change the formatting of the date variables (`pickup_datetime` and `dropoff_datetime`). This will help a lot with data extraction in the coming section.</p>

# In[ ]:


train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime) #Not in Test


# 1. <h2>3. Data Visualisation and Analysis</h2>
# ----------------------------------
# <p> These next steps involve looking at the data visually. Often you'll discover looking at something significant as a graph rather than a table (for example) will give you far greater insight into its nature and what you might need to do to work with it. Of course, the opposite could also be considered true, so don't neglect the first section we went through.</p>
# 
# <h3>3.1 Initial Analysis</h3><br>
# 
# <p> Let's plot a simple histogram of the trip duration, throwing the data into 100 bins. Change this around to get a feel for what binning does to your data. Simply put, binning involves taking your data's max and min points, subtracting it to get the length, dividing that length by the number of bins to get the interval length, and grouping the data points into those intervals. Here's what that looks like:</p> 

# In[ ]:


plt.hist(train['trip_duration'].values, bins=100)
plt.xlabel('trip_duration')
plt.ylabel('number of train records')
plt.show()


# <p>This is a good opportunity to play with some data transformations to see if notable patterns emerge in the data when applying certain transforms, for example a log transform. In this case, applying a log transformation to the trip duration makes sense, since we are doing this to accommodate the leaderboard's scoring metric. That would look like this:</p>

# In[ ]:


train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)
plt.hist(train['log_trip_duration'].values, bins=100)
plt.xlabel('log(trip_duration)')
plt.ylabel('number of train records')
plt.show()
sns.distplot(train["log_trip_duration"], bins =100)


# <p>One might also be interested to view the number of trips over time, since this could reveal not only apparent seasonality in the data and certain trends, but could point out any siginficant outliers (if not already cleaned out of the data set of course) and indicate missing values (again, only if it wasn't already checked and cleaned in the data set).</p>
# 
# <p>For this we'll simply plot a timeseries line graph of both the test and training data to not only look into identifying possible trends/seasonality but to see if both data sets follow the same pattern shape. Reasonably we'd expect the two datasets to follow a very similar shape since the test data would be/is a randomly selected sample from the original dataset containing all data points. By picking the test points randomly, each data point has the same likelihood of being picked as a test point, thus ensuring uniformity in the test data sample.</p>

# In[ ]:


plt.plot(train.groupby('pickup_date').count()[['id']], 'o-', label='train')
plt.plot(test.groupby('pickup_date').count()[['id']], 'o-', label='test')
plt.title('Trips over Time.')
plt.legend(loc=0)
plt.ylabel('Trips')
plt.show()


# <p>Clearly the test and train datasets follow a very simila shape, as expected. A couple of points stand out at first glance. Around late-Jan/early-Feb there is a massive drop in the number of trips taken. A slightly less drastic drop is apparent about four months later. The first drop could be related to the season: it's winter in New York so you'd expect less trips being taken (who wants to ride around when it's near freezing outside?). However, this seems unlikely because the dip looks to be isolated around a single day or couple of days. In my opinion there's a greater chance that there were strikes (if you've got the South African mindset I do, but this is New York, so probably not that) or that there was an issue with the data system recording the trips. Whatever the reason, it's worth noting these 'outliers'.</p>
# 
# <p> Let's see how significantly (or not significantly at all) the two vendors differ in their respective mean trip durations:</p>

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
plot_vendor = train.groupby('vendor_id')['trip_duration'].mean()
plt.subplots(1,1,figsize=(17,10))
plt.ylim(ymin=800)
plt.ylim(ymax=840)
sns.barplot(plot_vendor.index,plot_vendor.values)
plt.title('Time per Vendor')
plt.legend(loc=0)
plt.ylabel('Time in Seconds')


# <p>So it doesn't look like there's much of a difference between travel times for the two vendors. One would assume that knowing which routes are the fastest to take from A to B is no secret, and that it is more a trick of the trade, rather than IP. But something doesn't look right, so there's another feature we can use to see if after all there is a significant difference in mean travel time: the `store_and_fwd_flag`.</p>

# In[ ]:


snwflag = train.groupby('store_and_fwd_flag')['trip_duration'].mean()

plt.subplots(1,1,figsize=(17,10))
plt.ylim(ymin=0)
plt.ylim(ymax=1100)
plt.title('Time per store_and_fwd_flag')
plt.legend(loc=0)
plt.ylabel('Time in Seconds')
sns.barplot(snwflag.index,snwflag.values)


# <p>So it would seem that the `store_and_fwd_flag` discriminates well between travel times. Clearly there is a slight skew in the data where some of the vendor employees didn't record their travel times accurately.</p>
# 
# <p>As mentioned earlier when digging into the variables, I thought about the impact that the number of passengers per trip might have on travel time. The thought process was that more passengers might equal more stops, hence prolonging the travel time from start to finish (unless the different drop-offs were recorded as separate trips). So in order to find out if there is infact a significant influence in travel time, let's group the mean travel times by the `passenger_count`:</p>

# In[ ]:


pc = train.groupby('passenger_count')['trip_duration'].mean()

plt.subplots(1,1,figsize=(17,10))
plt.ylim(ymin=0)
plt.ylim(ymax=1100)
plt.title('Time per store_and_fwd_flag')
plt.legend(loc=0)
plt.ylabel('Time in Seconds')
sns.barplot(pc.index,pc.values)


# <p>So no significant difference evident that could be explained by the number of passengers in the vehicle for any given trip. It is interesting to note that there is are on average +-4min trips associated with no passengers. This is probably a mistake made in recording the data unless the vendor employee is into charging him/herself for trips whilst on the job.</p>
# 
# <p>Again, we need to check whether the test and train data matches with respect to the number of trips containing X-number of passengers:</p>

# In[ ]:


train.groupby('passenger_count').size()


# In[ ]:


test.groupby('passenger_count').size()


# <p>At first glance nothing seems too far out of the ordinary, however we can note that the passenger count is not ordinal. Note the two extra trips with 9 passengers in the test data. Again, it would seem there is some outlier data we haven't dealt with yet. Other than that though, the counts do seem to match relatively between the two data sets. The next step is to dive into the latitude and longitude data provided, trying to visualise the routes taken and potentially creating a map view to get a feel for the spacial components associated with the data.</p>

# ![](http://)<h3>3.2 Coordinate Mapping</h3><br>
# 
# <p>Just like we compared the travel time data and the number of passengers between the test and train sets, we can try and verify that the pickup location data in both sets are fairly similar and representative of one another.</p><br> 
# 
# <h3>3.2.1 Pickup Locations</h3><br>
# <p>To do this, we utilise the city map border coordinates for New York, mentioned earlier in the kernel to create the canvas wherein the coordinate points will be graphed. To display the actual coordinates a simple scatter plot is used:</p>

# In[ ]:


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(train['pickup_longitude'].values[:100000], train['pickup_latitude'].values[:100000],
              color='blue', s=1, label='train', alpha=0.1)
ax[1].scatter(test['pickup_longitude'].values[:100000], test['pickup_latitude'].values[:100000],
              color='green', s=1, label='test', alpha=0.1)
fig.suptitle('Train and test area complete overlap.')
ax[0].legend(loc=0)
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[1].set_xlabel('longitude')
ax[1].legend(loc=0)
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


# > <p>We can tell from the two graphs that the pickup locations are quite similar, with the notable difference being that the train data set simply has more data points (which makes sense).</p>
# 
# <h3>3.2.2 Distance and Directionality</h3><br>
# <p>This next part is quite interesting. Thanks to Beluga's post, we can determine the distance and direction of a specific trip based on the pickup and dropoff coordinates. For this I've made three functions, as:</p>

# In[ ]:


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


# <p>Applying these functions to both the test and train data, we can calculate the haversine distance which is the great-circle distance between two points on a sphere given their longitudes and latitudes. We can then calculate the summed distance traveled in Manhattan. And finally we calculate (through some handy trigonometry) the direction (or bearing) of the distance traveled. These calculations are stored as variables in the separate data sets. The next step I decided to take is to create neighourhods, like Soho, or the Upper East Side, from the data and display this.</p>

# In[ ]:


train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)    
    
train.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] =  dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)


# <h3>3.3.3 Lets create the "Neighborhoods"</h3><br>
# <p>One might think it necessary to have a map handy to do this, but not really. This will intuitively work as KMeans will cluster the data points into their own neighborhoods. This is pretty straight forward since Numpy helps create a vertically stacked array of the pickup and dropoff coordinates, and using `sklearn`'s MiniBatchKMeans module it's easy to set up the parameters to create the clusters.</p>
# 
# <p>There are three steps to preparing the data: create the coordinates stack, configure the KMeans clustering parameters, and create the actual clusters:</p>

# In[ ]:


coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))


# In[ ]:


sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])


# In[ ]:


train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])


# </p>Once you've run through these steps, you're all set to display these clusters in a similar way than the pickup locations. So as an example, I'll graph the pickup location clusters. Again, the map is drawn on the New York border coordinates canvas we created earlier:</p>

# In[ ]:


fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(train.pickup_longitude.values[:500000], train.pickup_latitude.values[:500000], s=10, lw=0,
           c=train.pickup_cluster[:500000].values, cmap='autumn', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# <p>This shows a nice visual representation of the KMeans clustering algorithm at work (we used 100 clusters, but there is freedom to play around with this parameter to see how it changes the results). The clustering effectively created the different neighbourhoods in Manhattan as displayed by the borders between the different colors. This should be intuitive to some degree as a trip would differ from point A to point B, in various parts of New york. By nature, it's different.</p>
# 
# <p>The next step is to start looking into extracting information from the dates and starting to think about encoding the data.</p> 

# <h3>3.4 Date Extraction</h3><br>
# 
# <p>Part of the reasoning behind extracting the different parts of the date for each trip is to enable us to do one hot encoding. This involves changing categorical variables into binary variables. It makes it easier to use when training ML models, since logically a machine can much better understand 1's and 0's rather than 'January' or 'Febuary', for example. You can read more on one hot encoding [here](http://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/ "One Hot Encoding"). To make sure we can use the features we extract from the dates, we need to check if both data sets has the same size (i.e. same number of months, days, hours, etc.). </p>

# In[ ]:


#Extracting Month
train['Month'] = train['pickup_datetime'].dt.month
test['Month'] = test['pickup_datetime'].dt.month


# In[ ]:


train.groupby('Month').size(),test.groupby('Month').size()


# <p>Both train and test data sets have 6 months, so months are ready for dummy variables (i.e. encoding).</p>

# In[ ]:


train['DayofMonth'] = train['pickup_datetime'].dt.day
test['DayofMonth'] = test['pickup_datetime'].dt.day
len(train.groupby('DayofMonth').size()),len(test.groupby('DayofMonth').size())


# <p>Both train and test data sets have 31 days, so days are ready for dummy variables (i.e. encoding).</p>

# In[ ]:


train['Hour'] = train['pickup_datetime'].dt.hour
test['Hour'] = test['pickup_datetime'].dt.hour
len(train.groupby('Hour').size()),len(test.groupby('Hour').size())


# <p>Both train and test data sets have 24 hours, so hours are ready for dummy variables (i.e. encoding).</p>

# In[ ]:


train['dayofweek'] = train['pickup_datetime'].dt.dayofweek
test['dayofweek'] = test['pickup_datetime'].dt.dayofweek
len(train.groupby('dayofweek').size()),len(test.groupby('dayofweek').size())


# Both sets of data has 7 days per week. That's fine for encoding aswell.

# <p>So, excellent! We can safely use the different date parts in their extracted forms as part of the modelling process. Let's take a look at the average speed and how it changes over time, specifically focusing on how the hour of the day, the day of the week, and the moth of the year affects average speed. It's important to note though that average speed is a function of distance and time so it wouldn't add anything to the modelling output. We'll therefore need to remove it eventually before we train our model.</p>

# In[ ]:


train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']
fig, ax = plt.subplots(ncols=3, sharey=True)
ax[0].plot(train.groupby('Hour').mean()['avg_speed_h'], 'bo-', lw=2, alpha=0.7)
ax[1].plot(train.groupby('dayofweek').mean()['avg_speed_h'], 'go-', lw=2, alpha=0.7)
ax[2].plot(train.groupby('Month').mean()['avg_speed_h'], 'ro-', lw=2, alpha=0.7)
ax[0].set_xlabel('Hour of Day')
ax[1].set_xlabel('Day of Week')
ax[2].set_xlabel('Month of Year')
ax[0].set_ylabel('Average Speed')
fig.suptitle('Average Traffic Speed by Date-part')
plt.show()


# <p>So the interesting thing to notice here is the average speed by hour of day. We're I'm from traffic usually peaks between 5am and 9am, and then again from about 4pm to around 6 or 7pm. But it would seem in manhattan that average speed diminishes as the day goes by from around 6am and picks up again around 7 or 8pm. So most of the travelling in the Big Apple happens during work hours. The average speed by weekday follows an expected trend. Over the weekend (Friday, Saturday, Sunday) the average speed picks up quite nicely, indicating less traffic. Finally, the average trip speed by month follows an expected trend. In the winter months there are less trips (see the previous timeseries plot we made) indicating less traffic in general in the city which means you can average a higher speed on the roads.</p>
# <br>
# 
# <p>This next part uses the pick-up locations and the average speed data we've got and plots the average speed by location. </p>

# In[ ]:


train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 3)
train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 3)
# Average speed for regions
gby_cols = ['pickup_lat_bin', 'pickup_long_bin']
coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()
coord_count = train.groupby(gby_cols).count()[['id']].reset_index()
coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)
coord_stats = coord_stats[coord_stats['id'] > 100]
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(train.pickup_longitude.values[:500000], train.pickup_latitude.values[:500000], color='black', s=1, alpha=0.5)
ax.scatter(coord_stats.pickup_long_bin.values, coord_stats.pickup_lat_bin.values, c=coord_stats.avg_speed_h.values,
           cmap='RdYlGn', s=20, alpha=0.5, vmin=1, vmax=8)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.title('Average speed')
plt.show()


# <p>So clearly, by neighbourhood, the average speed definitely changes. To a greater extent the center of the city is the busiest (we'd expect this since the majority of activity in large cities is focused around the center) and the average speed picks up nicely around the outskirts.</p><br>
# 
# <p>We can expect good performance from our clustering data during modelling just from looking at how well we can distinguish average speeds by neighbourhood (i.e. cluster). Something definitely worth exploring, which could boost the performance of the XGBoost model significantly, is to create a data set that can be used with [Xiaolin Wu's line algorithm](https://en.m.wikipedia.org/wiki/Xiaolin_Wu%27s_line_algorithm "Xiaolin Wu's line algorithm"). This would involve pixelating the graph area and recording every pixel crossed by the line from the pick-up location to the drop-off location. If you can make the resolution as high as possible some of the pixels shoudl encapsulate traffic junctions, traffic lights, bridges, etc. Using the "has crossed coordinate X" features one could potentially create an extra +-10 000 features to train the alogrithm with.</p><br>
# 
# <p>**Note:** XGBoost will be able to handle up to about 10 000 columns (features) with about 1 million rows of data on a Mac Book Pro.</p>

# <h2>4. Data Enrichment & Dummy Variables</h2>
# ---------------------------------------
# <p>At this point it's possible to train our model, but often in Data Science and analysing different datasets, we need to look for alternative sources of data that will add accuracy to the models we build. And we still have the opportunity to create dummy variables that can add to model's accuracy.</p><br>
# 
# <h3>4.1 Data Enrichment</h3><br>
# <p>For this particular problem we can add OSRM ([Open Source Routing Machine](http://project-osrm.org/ "OSRM")) features. Ultimately this is a data set containing the fastest routes from specific starting points in NY. </p>

# In[ ]:


fr1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps'])
fr2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
test_street_info = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv',
                               usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_street_info = pd.concat((fr1, fr2))
train = train.merge(train_street_info, how='left', on='id')
test = test.merge(test_street_info, how='left', on='id')


# In[ ]:


train.shape, test.shape


# <p>Strictly speaking we don't need to check the content/shape of the data, but for sanity we can check what the test and training data sets look like. The output makes sence: The columns that the train dataset has that test doesn't is now: `dropoff_datetime`, `avg_speed_m`, `avg_speed_h`, `pickup_lat_bin`, and `pickup_long_bin` (apart from `trip_duration`).</p><br>
# 
# <p>Now for the scary part (not really): Creating the dummy variables.</p><br>
# 
# <h3>4.2 Creating Dummy Variables</h3><br>
# <p>So for this step we get to the one hot encoding we spoke of earlier. Generally speaking you can do this in a few ways, but luckily Pandas helps us out again. A simple function that changes categorical data into dummy/indicator variables. Read more [here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html "Pandas get_dummies"). The application is simple:</p>

# In[ ]:


vendor_train = pd.get_dummies(train['vendor_id'], prefix='vi', prefix_sep='_')
vendor_test = pd.get_dummies(test['vendor_id'], prefix='vi', prefix_sep='_')
passenger_count_train = pd.get_dummies(train['passenger_count'], prefix='pc', prefix_sep='_')
passenger_count_test = pd.get_dummies(test['passenger_count'], prefix='pc', prefix_sep='_')
store_and_fwd_flag_train = pd.get_dummies(train['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
store_and_fwd_flag_test = pd.get_dummies(test['store_and_fwd_flag'], prefix='sf', prefix_sep='_')
cluster_pickup_train = pd.get_dummies(train['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_pickup_test = pd.get_dummies(test['pickup_cluster'], prefix='p', prefix_sep='_')
cluster_dropoff_train = pd.get_dummies(train['dropoff_cluster'], prefix='d', prefix_sep='_')
cluster_dropoff_test = pd.get_dummies(test['dropoff_cluster'], prefix='d', prefix_sep='_')

month_train = pd.get_dummies(train['Month'], prefix='m', prefix_sep='_')
month_test = pd.get_dummies(test['Month'], prefix='m', prefix_sep='_')
dom_train = pd.get_dummies(train['DayofMonth'], prefix='dom', prefix_sep='_')
dom_test = pd.get_dummies(test['DayofMonth'], prefix='dom', prefix_sep='_')
hour_train = pd.get_dummies(train['Hour'], prefix='h', prefix_sep='_')
hour_test = pd.get_dummies(test['Hour'], prefix='h', prefix_sep='_')
dow_train = pd.get_dummies(train['dayofweek'], prefix='dow', prefix_sep='_')
dow_test = pd.get_dummies(test['dayofweek'], prefix='dow', prefix_sep='_')


# <p>See how simple that was?</p><br>
# 
# <p>We do need to check the output though (never fully trust functions without testing it).</p>

# In[ ]:


vendor_train.shape,vendor_test.shape


# In[ ]:


passenger_count_train.shape,passenger_count_test.shape


# In[ ]:


store_and_fwd_flag_train.shape,store_and_fwd_flag_test.shape


# In[ ]:


cluster_pickup_train.shape,cluster_pickup_test.shape


# In[ ]:


cluster_dropoff_train.shape,cluster_dropoff_test.shape


# In[ ]:


month_train.shape,month_test.shape


# In[ ]:


dom_train.shape,dom_test.shape


# In[ ]:


hour_train.shape,hour_test.shape


# In[ ]:


dow_train.shape,dow_test.shape


# <p>Looking at the output you can tell that everything matches nicely, except for passenger count. We knew this however, since  there were two trips with 9 passengers in the test set where there was no trips with 9 passengers in the train set. Like we concluded, this is clearly an error/outlier. Let's get rid of that:</p>

# In[ ]:


passenger_count_test = passenger_count_test.drop('pc_9', axis = 1)


# <p>And it would seem we've finally reached the end of what is easily considered the most difficult part of data science: cleaning and wrangling the data into a usable format/structure. The last part before we actually start training the model is to drop the categorical variables (since we replaced these with our indicator variables in the previous step) and compile the final version of our test and train sets.</p>

# In[ ]:


train = train.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','Month','DayofMonth','Hour','dayofweek',
                   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'],axis = 1)
Test_id = test['id']
test = test.drop(['id','vendor_id','passenger_count','store_and_fwd_flag','Month','DayofMonth','Hour','dayofweek',
                   'pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'], axis = 1)

train = train.drop(['dropoff_datetime','avg_speed_h','avg_speed_m','pickup_lat_bin','pickup_long_bin','trip_duration'], axis = 1)


# <p>Let's check what's left after dropping our categorical variables.</p>

# In[ ]:


train.shape,test.shape


# <p>Now let's add the indicator variables to our datasets.</p>

# In[ ]:


Train_Master = pd.concat([train,
                          vendor_train,
                          passenger_count_train,
                          store_and_fwd_flag_train,
                          cluster_pickup_train,
                          cluster_dropoff_train,
                         month_train,
                         dom_train,
                          hour_test,
                          dow_train
                         ], axis=1)


# In[ ]:


Test_master = pd.concat([test, 
                         vendor_test,
                         passenger_count_test,
                         store_and_fwd_flag_test,
                         cluster_pickup_test,
                         cluster_dropoff_test,
                         month_test,
                         dom_test,
                          hour_test,
                          dow_test], axis=1)


# In[ ]:


Train_Master.shape,Test_master.shape


# <p>There're two more columns we can drop because we keep the information from those two columns in other variables: `pickup_datetime` and `pickup_date`.</p>

# In[ ]:


Train_Master = Train_Master.drop(['pickup_datetime','pickup_date'],axis = 1)
Test_master = Test_master.drop(['pickup_datetime','pickup_date'],axis = 1)


# <p>One more sense check.</p>

# In[ ]:


Train_Master.shape,Test_master.shape


# <p>This is exactly what we're expecting to see. The training set should have one more column than the test set, since it contains our target variable (which the test set doesn't). The next step is to split the training set into sub-training and sub-testing sets. The reason for this is to be able to tweak model parameters to increase accuracy (i.e. decrease the root mean square error [RSME] value) without creating bias towards the test set. Also, and perhaps more importantly, is that we need a validation set to use with XGBoost. So the XGBoost algorithm takes three datasets: a training set, a test set, and a validation set. The validation set is used to continuously evaluate the accuracy of the training model, and finally the model is used to make predictions against the test set. So splitting out the training set into a separate train and test set gives us a test sample of which we know the outcome variables. Makes sense, right?</p> <br>
# 
# <p>For the sake of this tutorial we'll stick to 100 000 data points, this just makes the training process faster and a little less cumbersome to work with. Also, we'll split the training set into a 80-20 train and test set. So from this, there are two things you can do: change the ratios by which you split the original training set (`Train_Master`), e.g. use a 70-30 ratio and see what the results are. Also, you can include all the data points instead of just 100k. This should also have quite a siginficant effect on the outcome.</p><br>
# 
# <p>So the split would look something like this:</p>

# In[ ]:


Train, Test = train_test_split(Train_Master[0:100000], test_size = 0.2)


# <p>Some additional steps are required for us to use these datasets. One thing is to drop the `log_trip_duration` features (this is merely a log transform of one of the features, so it's signficance will remain the same as when it isn't transformed). Also, we can drop our original index and reset it, to make sure we can reference the rows accurately if needed.</p>

# In[ ]:


X_train = Train.drop(['log_trip_duration'], axis=1)
Y_train = Train["log_trip_duration"]
X_test = Test.drop(['log_trip_duration'], axis=1)
Y_test = Test["log_trip_duration"]

Y_test = Y_test.reset_index().drop('index',axis = 1)
Y_train = Y_train.reset_index().drop('index',axis = 1)


# <p>One more step and it really is time to start training: We need to create the XGBoost matrices that will be used to train the model using XGBoost. Note that we use the newly created test and train sets as our model training inputs (for train and validation) and finally use the master test set to make our predictions.</p>

# In[ ]:


dtrain = xgb.DMatrix(X_train, label=Y_train)
dvalid = xgb.DMatrix(X_test, label=Y_test)
dtest = xgb.DMatrix(Test_master)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


# <p>Finally! We're ready to use XGBoost. You'll notice that this part is actually quite short. But only in code. It can take quite a while to itterate through the different parameters and different values of these parameters to better the outcome of the model. Let's see what's next.</p>

# [](http://)<h2>5. XGBoost - Training the Model and Testing the Accuracy</h2>
# -------------------------------------------------------
# 
# <p>As mentioned you can play with the different parameters of the XGBoost algorithm to tweak the model's outcome. So below is a short, but very nice, way of itterating through model parameters to tweak the model. So it's implementation is simple: just uncomment the code and run the kernel. Again, refer to the [documentation for XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html "XGBoost Documentation") to understand what each parameter does and how it alters the behaviour of the training process.</p>

# In[ ]:


#md = [6]
#lr = [0.1,0.3]
#mcw = [20,25,30]
#for m in md:
#    for l in lr:
#        for n in mcw:
#            t0 = datetime.now()
#            xgb_pars = {'min_child_weight': mcw, 'eta': lr, 'colsample_bytree': 0.9, 
#                        'max_depth': md,
#            'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
#            'eval_metric': 'rmse', 'objective': 'reg:linear'}
#            model = xgb.train(xgb_pars, dtrain, 50, watchlist, early_stopping_rounds=10,
#                  maximize=False, verbose_eval=1)


# [0]	train-rmse:5.70042	valid-rmse:5.69993
# Multiple eval metrics have been passed: 'valid-rmse' will be used for early stopping.
# 
# Will train until valid-rmse hasn't improved in 10 rounds.
# 
# *[11]	train-rmse: 3.25699	valid-rmse: 3.25698 <br>
# ... <br>
# ... <br>
# 
# [89]	train-rmse:0.335358	valid-rmse:0.345624 <br>
# [90]	train-rmse:0.334614	valid-rmse:0.344972 <br>
# [91]	train-rmse:0.333921	valid-rmse:0.344405 <br>

# Here is one for the fork's. I do recomend you trying to play with these metrics though... 

# In[ ]:


xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 
            'max_depth': 6,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}
model = xgb.train(xgb_pars, dtrain, 10, watchlist, early_stopping_rounds=2,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)


# <p>Of course, what type of tutorial would this be if I didn't give some ideas about where to twea the model? Here are some steps I suggest you follow:</p>
# <ol>
# <li>Make it more than 10 iterations before stopping.</li>
# <li>Lower the eta-value.</li>
# <li>Increase the max depth.</li>
# </ol>
# <p>Be very careful though, there is such a thing as overfitting. I.e. training your model so well on the training set that it performs quite poorly on unseen data. Here I would suggest playing with `colsample_bytree` and `subsample` to control overfitting.</p><br>
# 
# <p>Out of interest, we can investigate the importance of each feature, to understand what affects the trip duration the most significantly. Here's how XGBoost allows us to do it:</p>
# 

# In[ ]:


xgb.plot_importance(model, max_num_features=28, height=0.7)


# <p>So from top to bottom we see which features have the greatest affect on trip duration. It would make logical sense that distance has the greatest affect. The further you travel, the longer it'll take. The rest of the features follow a similar logic in why it's ranked the way it is.</p><br>
# 
# <p>The final step before submission is to make our predictions using the trained model:</p>

# In[ ]:


pred = model.predict(dtest)
pred = np.exp(pred) - 1


# <p>It's as simple as that. We've now successfully trained a model and made predictions on unseen data about how long it would take to get from A to B. Thus, the last step in getting yourself onto the leaderboard is to make a submission. Let's do that quickly:</p>

# Time for Submission

# In[ ]:


submission = pd.concat([Test_id, pd.DataFrame(pred)], axis=1)
submission.columns = ['id','trip_duration']
submission['trip_duration'] = submission.apply(lambda x : 1 if (x['trip_duration'] <= 0) else x['trip_duration'], axis = 1)
submission.to_csv("submission.csv", index=False)


# <p>And that's it folks. So thats my basic model on the NYC taxi problem. I truly hope everything was clear, and if not please feel free to comment and I'll try and clarify anything. Also if you spot some errors (come on, I'm human after all) please drop me a line and I'll correct it.</p><br>
# 
# <p>Thanks for forking, please share if it was useful, and please throw in an upvote if you think it deserves one.</p><br>
# 
# <b><p>Warm regards and happy kaggling</p><br></b>
# <h1>KarelRV</h1>
