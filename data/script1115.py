
# coding: utf-8

# *This kernel explores the [dataset of 2016 New York parties](https://www.kaggle.com/somesnm/partynyc). The data was extracted from party related noise complaints in the city. The whole dataset is available at New York Open Data portal.*
# 
# ## Introduction
# 
# Previous EDA's showed that the number of rides increases during the weekends, especially at night. This is expected because that is a prime time for going out and having fun. 
# 
# So I figured that finding the rides that take a passenger home from the party would help us in producing a better model.  Unfortunately, there is no database of all the best parties in the Big Apple, but during parties, people tend to get carried away and turn the music too loud which makes some neighbors unhappy (also probably because they weren't invited).  So they submit a complaint to a city hotline so the local police would check in on the party and ask the host to dial it down. Luckily the city of New York is very modern one and they embrace the Open Data movement and try to make all data freely available. 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import time


# In[ ]:


train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv',parse_dates=['pickup_datetime','dropoff_datetime'])
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',parse_dates=['pickup_datetime'])
party = pd.read_csv('../input/partynyc/party_in_nyc.csv',parse_dates=['Created Date','Closed Date'])


# In[ ]:


# Time between recieving the call and police arrival
party['ticket_duration'] = (party['Closed Date'].view('int64') // 10**9 - party['Created Date'].view('int64') // 10**9)/60
# Dropping all the tickets with negative duration
party = party[(party['ticket_duration']>0)&(party['ticket_duration']<24*60)]
party.dropna(inplace=True) # Getting rid of nan rows


# In[ ]:


#Additional features
party['hour'] = party['Created Date'].dt.hour
train['hour'] = train['pickup_datetime'].dt.hour
party['dayofweek'] = party['Created Date'].dt.weekday_name
party['date'] = party['Created Date'].dt.date
party['epoch'] = party['Created Date'].view('int64') // 10**9 # Unix time
train['epoch'] = train['pickup_datetime'].view('int64') // 10**9
test['epoch'] = test['pickup_datetime'].view('int64') // 10**9


# ### Columns: <br />
# **Created Date** - time of the call <br />
# **Closed Date** - time when ticket was closed by police <br />
# **Location Type** - type of the location <br />
# **Incident Zip** - zip code of the location <br />
# **City** - name of the city (almost the same as the Borough field) <br />
# **Borough** - administrative division of the city <br />
# **Latitude** - latitude of the location <br />
# **Longitude** - longitude of the location <br />

# In[ ]:


party.head()


# ### Brooklyn and Manhattan are epicenters of night life in the city

# In[ ]:


party['Borough'].value_counts(ascending=True).plot(kind='barh',title='Nuber of calls by district',
                                                   figsize=(8,5));


# ## Let's check how fast police responds to the complaint

# In[ ]:


party['ticket_duration'].plot.hist(title='Police response time in minutes',bins=20, figsize=(10,6));


# ### Response time by city district

# In[ ]:


party['ticket_duration'].hist(by=party['Borough'],figsize = (15,10),bins=20);


# In[ ]:


party['ticket_duration'].describe()


# ### Police respond quickest in Manhattan, but the median response time of almost 2.5 hours indicates that maybe opened tickets closed not when the police arrive, but at the end of the shift, so it's better to rely on the time of the call.

# ## Let's check the distribution of calls by weekdays, hours of the day and weeks of the year

# In[ ]:


wds = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
party['dayofweek'].value_counts().reindex(wds).plot(kind='bar',title="Number of calls by weekday",figsize=(10,6));


# In[ ]:


party['hour'].value_counts(sort=False).plot(kind='bar',title="Number of calls by hour of the day",figsize=(10,6));


# In[ ]:


party['date'].value_counts().sort_index().plot(figsize=(12,6),
                                               title='Number of calls by day of the year');


# ## No surprises here, the majority of parties occur on the weekends and during night hours. And the last plot shows the cyclic nature of the data, peaks represent the weekends. The beginning of the summer and beginning of the fall have the highest number of complaints, surprisingly the number of calls drops in August. 

# In[ ]:


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
#fig, ax = plt.subplots(ncols=1, sharex=True, sharey=True)
ax = plt.scatter(party['Longitude'].values, party['Latitude'].values,
              color='blue', s=0.5, label='train', alpha=0.1)
ax.axes.set_title('Coordinates of the calls')
ax.figure.set_size_inches(6,5)
plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()


# # Let's try to match the rides from training and test data to the parties near the beginning of the ride. 

# ## This function takes the taxi ride and returns the number of complaints that were submitted around pickup place (radius ~500 meters) in a time period from 2 hours before to 2 hours after the pickup time.

# In[ ]:


def match_party(row): 
    lat_range = 0.003 # 333 meters by latitude
    long_range = 0.004 ## 336 meters by longitude
    time_radius = 2*60*60 # 2 hours in seconds
    sl = party[party['epoch'].between(row['epoch']-time_radius,row['epoch']+time_radius)]
    sl = sl[sl['Longitude'].between(row['pickup_longitude']-long_range,row['pickup_longitude']+long_range)]
    sl = sl[sl['Latitude'].between(row['pickup_latitude']-lat_range,row['pickup_latitude']+lat_range)]
    return sl.shape[0]


# ### This code took me 4 hours to finish, so I added the result to the dataset, but you can reuse the code on your if you want to change time range or radius

# train['num_complaints'] = train.apply(lambda x: match_party(x),axis=1)
# train[['id','num_complaints']].to_csv('train_parties.csv',index=False)
# test['num_complaints'] = test.apply(lambda x: match_party(x),axis=1)
# test[['id','num_complaints']].to_csv('test_parties.csv',index=False)

# ### Importing the results of above code.

# In[ ]:


train['num_complaints'] = pd.read_csv('../input/partynyc/train_parties.csv')['num_complaints']
test['num_complaints'] = pd.read_csv('../input/partynyc/test_parties.csv')['num_complaints']


# ### 13.1% of all rides have at least one noise complaint nearby

# In[ ]:


train['num_complaints'].value_counts(normalize=True).head(10)


# ### 23.9% of the rides that occurred from 18 pm to 6 am have at least one noise complaint nearby

# In[ ]:


train[(train['hour']>18) | (train['hour']<6)]['num_complaints'].value_counts(normalize=True).head(10)


# In[ ]:


pd.read_csv('../input/partynyc/train_parties.csv').to_csv('train_parties.csv',index=False)
pd.read_csv('../input/partynyc/test_parties.csv').to_csv('test_parties.csv',index=False)

