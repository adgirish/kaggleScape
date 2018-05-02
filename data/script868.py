
# coding: utf-8

# This notebook is trying to create new features from the features given

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train = pd.read_json("../input/train.json")
test = pd.read_json("../input/train.json")


# In[ ]:


train['source']='train'
test['source']='test'
print(train.head())
print(test.info())


# Handling Bathrooms and bedrooms

# In[ ]:


print(train['bathrooms'].value_counts())
# There are houses with 0.0 bathrooms and some with floating point no of bathrooms
print(train['bedrooms'].value_counts())
# There are 9475 houses with 0 bedrooms
train['bathrooms']=train['bathrooms'].astype(int)
test['bathrooms']=test['bathrooms'].astype(int)
# if no of bathrooms are greater than 5 interest level is low else varies
# so we can create dummies
train.loc[train['bathrooms']==0, 'interest_level'] = 'low'
sns.violinplot(x='interest_level', y='bathrooms', data=train)
plt.xlabel('Interest level')
plt.ylabel('bathrooms')
plt.show()
# if 0 or greater than 4 bathrooms interest_level is low


# Creating new variables from bathrooms

# In[ ]:


train["price_tt"] =train["price"]/train["bathrooms"]
test["price_tt"] = test["price"]/test["bathrooms"] 

train["room_sum"] = train["bedrooms"]+train["bathrooms"] 
train["room_sum"] = test["bedrooms"]+test["bathrooms"] 


# Bedrooms

# In[ ]:


# same patern as bathrooms only different at 0
# pattern between bedrooms and  bathrooms
print(train.loc[train['bathrooms']==0, 'bedrooms'].value_counts())
print(train.loc[train['bathrooms']==6, 'bedrooms'].value_counts())
# we can combine 6 or more #bathrooms with 5 or more bedrooms
sns.violinplot(x='interest_level', y='bedrooms', data=train)
plt.xlabel('Interest level')
plt.ylabel('bedrooms')
plt.show()


# Adding variables from bedrooms

# In[ ]:


train["price_t"] =train["price"]/train["bedrooms"]
test["price_t"] = test["price"]/test["bedrooms"] 

train['room_sum'] = train['bedrooms']  + train['bathrooms']
test['room_sum'] = test['bedrooms']  + test['bathrooms']
train['price_per_room'] = train['price']/train['room_sum']
test['price_per_room'] = test['price']/test['room_sum']


# Handling time (created)

# In[ ]:


train["created"] = pd.to_datetime(train["created"])
train["created_year"] = train["created"].dt.year
train["created_month"] = train["created"].dt.month
train["created_day"] = train["created"].dt.day
train["created_hour"] = train["created"].dt.hour
test["created"] = pd.to_datetime(test["created"])
test["created_year"] = test["created"].dt.year
test["created_month"] = test["created"].dt.month
test["created_day"] = test["created"].dt.day
test["created_hour"] = test["created"].dt.hour


# Log of Price as it is right skewed

# In[ ]:


plt.scatter(range(train.shape[0]), np.sort(train.price.values))
plt.xlabel('index')
plt.ylabel('price')
plt.show()
# there are outliners
ulimit = np.percentile(train.price.values, 99)
train['price'].ix[train['price']>ulimit] = ulimit
# price is right skewed so using log to create a gaussian pattern
train['price']=np.log1p(train['price'])
test['price']=np.log1p(test['price'])

plt.figure(figsize=(8,6))
sns.distplot(train.price.values, bins=50, kde=True)
plt.xlabel('price')
plt.show()
sns.violinplot(data=train,x = 'interest_level',y='price')
plt.show()


# Street address and Display address

# In[ ]:


from sklearn.preprocessing import LabelEncoder
display_count = train.groupby('display_address')['display_address'].count()
plt.hist(display_count.values, bins=100, log=True, alpha=0.9)
plt.xlabel('Number of times display_address appeared', fontsize=12)
plt.ylabel('log of Count', fontsize=12)
plt.show()
# there are too many values and none of them are more than 500
# most of the values are less than 10
#so we label encode the values
address = ["display_address", "street_address"]
for x in address:
    le = LabelEncoder()
    le.fit(list(train[x].values))
    train[x] = le.transform(list(train[x].values))
    le.fit(list(test[x].values))
    test[x] = le.transform(list(test[x].values))
    


# Find position and neighbourhood from latitude and longitude

# In[ ]:


train["pos"] = train.longitude.round(3).astype(str) + '_' + train.latitude.round(3).astype(str)
test["pos"] = test.longitude.round(3).astype(str) + '_' + test.latitude.round(3).astype(str)

from sklearn.cluster import Birch
def cluster_latlon(n_clusters, data):  
    print("data.shape")
    print(data.shape)
    #split the data between "around NYC" and "other locations" basically our first two clusters 
    data_c=data[(data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)]
    print(data_c.shape)
    data_e=data[~((data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9))]
    print(data_e.shape)
    #put it in matrix form
    coords=data_c.as_matrix(columns=['latitude', "longitude"])
    
    brc = Birch(branching_factor=100, n_clusters=n_clusters, threshold=0.01,compute_labels=True)

    brc.fit(coords)
    clusters=brc.predict(coords)
    data_c["cluster_"+str(n_clusters)]=clusters
    data_e["cluster_"+str(n_clusters)]=-1 #assign cluster label -1 for the non NYC listings 
    data=pd.concat([data_c,data_e])
    print(data.shape)
    #plt.scatter(data_c["longitude"], data_c["latitude"], c=data_c["cluster_"+str(n_clusters)], s=10, linewidth=0.1)
    #plt.title(str(n_clusters)+" Neighbourhoods from clustering")
    #plt.show()
    return data 

train=cluster_latlon(100, train)
print(train.head())
clusters_price_map=dict(train.groupby(by="cluster_100")["price"].median())
train["price_comparison"]=train['price']-train["cluster_100"].map(clusters_price_map)

test=cluster_latlon(100, test)

clusters_price_map=dict(test.groupby(by="cluster_100")["price"].median())
test["price_comparison"]=test['price']-test["cluster_100"].map(clusters_price_map)


# manager_id

# In[ ]:


print(len(train['manager_id'].unique()))
# 3481 unique managers
temp = train.groupby('manager_id').count().iloc[:,-1]
temp2 = test.groupby('manager_id').count().iloc[:,-1]
train_managers = pd.concat([temp,temp2],axis=1,join='outer')
train_managers.columns=['train_count','test_count']
print(train_managers.sort_values(by = 'train_count',ascending = False).head())
# considering only those manager_ids which are in train
man_list = train_managers['train_count'].sort_values(ascending = False).head(3481).index
ixes = train.manager_id.isin(man_list)
train10 = train[ixes][['manager_id','interest_level']]
# create dummies of interest levels
interest_dummies = pd.get_dummies(train10.interest_level)
train10 = pd.concat([train10,interest_dummies[['low','medium','high']]], axis = 1).drop('interest_level', axis = 1)
print(train10.head())
gby = pd.concat([train10.groupby('manager_id').mean(),train10.groupby('manager_id').count()], axis = 1).iloc[:,:-2]
gby.columns = ['low','medium','high','count']
gby.sort_values(by = 'count', ascending = False).head(10)
gby['manager_skill'] = gby['medium']*1 + gby['high']*2 
gby['manager_id']=gby.index
print(gby.head(5))
print(gby.shape)
train = train.merge(gby[['manager_id','manager_skill']],on='manager_id',how='outer',right_index=False)
train['manager_skill']=train['manager_skill'].fillna(0)
print(train.head())


# Adding interest acc to manager_id to test acc to train data

# In[ ]:


index=list(range(train.shape[0]))
random.shuffle(index)
a=[np.nan]*len(train)
b=[np.nan]*len(train)
c=[np.nan]*len(train)

for i in range(5):
    building_level={}
    for j in train['manager_id'].values:
        building_level[j]=[0,0,0]
    
    test_index=index[int((i*train.shape[0])/5):int(((i+1)*train.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    
    for j in train_index:
        temp=train.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=1
            
    for j in test_index:
        temp=train.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0/sum(building_level[temp['manager_id']])
            
train['manager_level_low']=a
train['manager_level_medium']=b
train['manager_level_high']=c

a=[]
b=[]
c=[]
building_level={}
for j in train['manager_id'].values:
    building_level[j]=[0,0,0]

for j in range(train.shape[0]):
    temp=train.iloc[j]
    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=1

for i in test['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0/sum(building_level[i]))
        b.append(building_level[i][1]*1.0/sum(building_level[i]))
        c.append(building_level[i][2]*1.0/sum(building_level[i]))
test['manager_level_low']=a
test['manager_level_medium']=b
test['manager_level_high']=c
print(train.head())
print(test.head())


# Adding interest level  Building_id decreases the model accuracy

# Adding no of photos , no of words in features and description

# In[ ]:


train["num_photos"] = train["photos"].apply(len)
test["num_photos"] = test["photos"].apply(len)

train["num_features"] = train["features"].apply(len)
test["num_features"] = test["features"].apply(len)

train["num_description_words"] = train["description"].apply(lambda x: len(x.split(" ")))
test["num_description_words"] = test["description"].apply(lambda x: len(x.split(" ")))


# Changing features to sparse matrix

# In[ ]:


train['features'] = train["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test['features'] = test["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

tfidf = CountVectorizer(stop_words='english', max_features=200)
tr_sparse = tfidf.fit_transform(train["features"])
te_sparse = tfidf.transform(test["features"])

