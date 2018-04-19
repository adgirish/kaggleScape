
# coding: utf-8

# This kernel provides an overview of the KKBox  competition dataset  with some useful data transformations.

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


# ## Members

# First of all we are going to explore the Members dataset. We are dealing with 34403 members.

# In[ ]:


members = pd.read_csv('../input/members.csv')
members.head()


# In[ ]:


members.shape


# Here follows the distribution of the members ages. As you can see there are several with age set to 0. That is obsviously something we'll have to manage.

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

f,axarray = plt.subplots(1,1,figsize=(15,10))
agehist = members.groupby(['bd'],as_index=False).count()
sns.barplot(x=agehist['bd'],y=agehist['gender'])


# Members come from 21 different cities identified with an integer index ranging from 1 to 21. Here is the distribution of the members cities.

# In[ ]:


f,axarray = plt.subplots(1,1,figsize=(15,10))
cityhist = members.groupby(['city'],as_index=False).count()
sns.barplot(x=cityhist['city'],y=cityhist['gender'])


# ## Songs

# Now let's have a look at the Songs dataset.

# In[ ]:


songs = pd.read_csv('../input/songs.csv')
songs.head()


# There are 329825 composers ...

# In[ ]:


len(songs['composer'].unique())


# ... for 1046 genres ...

# In[ ]:


len(songs['genre_ids'].unique())


# and 11 different languages !

# In[ ]:


len(songs['language'].unique())


# ## Train Data

# In order to create the complete dataset we have to mearge the train, members and songs dataframe.

# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


df = train.merge(members,how='inner',on='msno')


# In[ ]:


df = df.merge(songs,how='inner',on='song_id')
df.head()


# In[ ]:


df.shape


# In[ ]:


cities = df['city'].unique()
ca = []
for c in cities:
    ages = []
    tmp = df[df['city']==c].groupby(['bd'],as_index=False).count()
    for i in range(60):
        if i in tmp['bd'].values:
            if i ==0:
                ages.append(0)
            else:
                ages.append(tmp[tmp['bd']==i].values[0][1])
        else:
            ages.append(0)
    ca.append(ages)
cadf = pd.DataFrame(ca)


# It may be interesting to see how the members ages are distributed among the 21 cities. Let's create a heatmap for that !

# In[ ]:


f,axarray = plt.subplots(1,1,figsize=(13,8))
sns.heatmap(cadf)


# In[ ]:


fdf = df[np.abs(df['bd']-df['bd'].mean())<=(3*df['bd'].std())]


# In[ ]:


cities = fdf['city'].unique()
ca = []
for c in cities:
    ages = []
    tmp = fdf[fdf['city']==c]['bd'].values
    ages.append(tmp)
    ca.append(ages)
cadf = pd.DataFrame(ca)


# In[ ]:


f,axarray = plt.subplots(21,1,figsize=(20,38),sharex=True)
plt.xlim(10,60)
for i in range(21):
    axarray[i].set_title('Members Ages in City '+str(i))
    sns.distplot(ca[i], hist=False, color="purple", kde_kws={"shade": True},ax=axarray[i])


# Now let's turn registration and expiration times into datetime types.

# In[ ]:


df['registration_init_time'] = pd.to_datetime(df['registration_init_time'],format="%Y%m%d")
df['expiration_date'] = pd.to_datetime(df['expiration_date'],format="%Y%m%d")


# In[ ]:


df.head()


# In[ ]:


days = df.expiration_date - df.registration_init_time
days = [d.days for d in days]
df['days']=days


# In[ ]:


np.max(days)


# This allows us to easily count the days of each membership.

# In[ ]:


df.head()


# Let's remove the outliers:

# In[ ]:


fdf = df[np.abs(df['days']-df['days'].mean())<=(3*df['days'].std())]


# In[ ]:


dayshist = df.groupby(['days'],as_index=False).count()
dayshist = dayshist.drop(0,axis=0)


# In[ ]:


sns.distplot(dayshist['days'], hist=True, color="g", kde_kws={"shade": True})


# In[ ]:


cities = fdf['city'].unique()
cduration = []
for c in cities:
    duration = []
    tmp = fdf[fdf['city']==c]['days']
    cduration.append(tmp)


# Let's analyze the distrubution of th  subscription durations among the 21 cities.

# In[ ]:


f,axarray = plt.subplots(21,1,figsize=(20,38),sharex=True)
for i in range(21):
    axarray[i].set_title('Subscription Durations in City '+str(i))
    sns.distplot(cduration[i], hist=False, color="g", kde_kws={"shade": True},ax=axarray[i])


# In[ ]:


malec = len(df[df['gender']=='male'])
femalec = len(df[df['gender']=='female'])


# Members are homogenously distributed between the two genders.

# In[ ]:


f,axarray = plt.subplots(1,1,figsize=(8,5))
sns.barplot(x=['male','female'],y=[malec,femalec])


# unforunately 40% of rows have NaN gender 

# In[ ]:


len(df[pd.isnull(df['gender'])])/len(df)


# There are 573 differen genres, let's see what are the most appreciated.

# In[ ]:


len(df['genre_ids'].unique())


# In[ ]:


ghist = df.groupby(['genre_ids'],as_index=False).count()


# In[ ]:


f,axa = plt.subplots(1,1, figsize=(12,18))
tghist = ghist[ghist['msno']>1000]
sns.barplot(y=tghist['genre_ids'],x=tghist['msno'],orient='h')


# ## Prediction

# target feature is the target variable. target=1 means there are recurring listening event(s) triggered within a month after the user’s very first observable listening event, target=0 otherwise .

# In[ ]:


df.head()


# In[ ]:


tmp= df.groupby(['msno'],as_index=False).count()['song_id']
tmp.describe()


# In[ ]:


f,axa = plt.subplots(1,1,figsize=(15,8))
sns.distplot(tmp.values)


# work in progress...
