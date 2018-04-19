
# coding: utf-8

# This an idea on how to use lat lon data to create a new variable. At first I wanted to use NYC's districts polygon map to derive a district appartenance for each listing_id but that would have been using external data. So instead of this I figured I could just derive natural district from the lat long data using a little clustering. 

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


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


train=pd.read_json("../input/train.json")
test=pd.read_json("../input/test.json")
train["Source"]='train'
test["Source"]='test'
data=pd.concat([train, test]) 


# In[ ]:


plt.scatter(data["longitude"], data["latitude"], s=5)
plt.title("Geographical positions of the listings")
plt.show()


# ## Geographic plotting ##

# Clearly, it seems that some of the data is missing (0,0): unless there are some people are looking for an appartment right in the guinea gulf.  Let's remove them so we can have a better view of the data. 

# In[ ]:


plt.scatter(data.loc[data["longitude"]<-60,"longitude"], data.loc[data["latitude"]>20,"latitude"], s=5)
plt.title("Geographical positions of the listings")
plt.show()


# There are a few flats all around the US but most of the cloud is around NYC
# So lets zoom in on NYC 

# In[ ]:


plt.scatter(data.loc[(data["longitude"]<-73.75)&(data["longitude"]>-74.05)&(data["latitude"]>40.4)&(data["latitude"]<40.9),"longitude"],
                      data.loc[(data["latitude"]>40.4)&(data["latitude"]<40.9)&(data["longitude"]<-73.75)&(data["longitude"]>-74.05),"latitude"], s=5)
plt.title("Geographical positions of the listings")
plt.show()


# At this level there are enough points to see some known features: we see the shape of manhattan with a hole for central park for example. 

# ## Clustering NYC data ##

# In[ ]:




#I use Birch because of how fast it is. 
from sklearn.cluster import Birch
def cluster_latlon(n_clusters, data):  
    #split the data between "around NYC" and "other locations" basically our first two clusters 
    data_c=data[(data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)]
    data_e=data[~(data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)]
    #put it in matrix form
    coords=data_c.as_matrix(columns=['latitude', "longitude"])
    
    brc = Birch(branching_factor=100, n_clusters=n_clusters, threshold=0.01,compute_labels=True)

    brc.fit(coords)
    clusters=brc.predict(coords)
    data_c["cluster_"+str(n_clusters)]=clusters
    data_e["cluster_"+str(n_clusters)]=-1 #assign cluster label -1 for the non NYC listings 
    data=pd.concat([data_c,data_e])
    plt.scatter(data_c["longitude"], data_c["latitude"], c=data_c["cluster_"+str(n_clusters)], s=10, linewidth=0.1)
    plt.title(str(n_clusters)+" Neighbourhoods from clustering")
    plt.show()
    return data 


# The ideal algorithm for this would be DBSCAN as shown in here http://geoffboeing.com/2014/08/clustering-to-reduce-spatial-data-set-size/ however it is too heavy to run on this kernel because I believe it has to compute a matrix of distance between each points. The advantage of DBSCAN is that it would leave the "extremum" point out of the clusters while Birch here is creating clusters of very low density for those 

# ## Naive variables##

# In[ ]:


data["created"]=pd.to_datetime(data["created"])
data["created_month"]=data["created"].dt.month
data["created_day"]=data["created"].dt.day
data["created_hour"]=data["created"].dt.hour


# In[ ]:


data["num_photos"]=data["photos"].apply(len)
data["num_features"]=data["features"].apply(len)
data["num_description_words"] = data["description"].apply(lambda x: len(x.split(" ")))


# In[ ]:


features_to_use_1  = ["bathrooms", "bedrooms", "price", 
                                                     
                    "num_photos", "num_features", "num_description_words",                    
                    "created_month", "created_day", "created_hour"
                   ]


# In[ ]:


def test_train(data, features):
    train=data[data["Source"]=="train"]
    test=data[data["Source"]=="test"]
    target_num_map={"high":0, "medium":1, "low":2}
    y=np.array(train["interest_level"].apply(lambda x: target_num_map[x]))
    from sklearn.model_selection import train_test_split
    X_train, X_val,y_train, y_val =train_test_split( train[features], y, test_size=0.33, random_state=42)
    return (X_train, X_val,y_train, y_val )


# In[ ]:


X_train, X_val,y_train, y_val=test_train(data, features_to_use_1)


# In[ ]:


from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier as RFC

def prediction(X_train,y_train, X_val, y_val):
    clf=RFC(n_estimators=1000, random_state=42)
    clf.fit(X_train, y_train)


    y_val_pred = clf.predict_proba(X_val)
    return(log_loss(y_val, y_val_pred))


# In[ ]:


prediction(X_train,y_train, X_val, y_val)


# ## Adding the cluster variable ##

# In[ ]:


from sklearn.metrics import log_loss

def compute_logloss(n_cluster,data):
    data_cluster=cluster_latlon(n_cluster,data)
      
    features = ["bathrooms", "bedrooms", "price", 
                                                        
                    "num_photos", "num_features", "num_description_words",                    
                    "created_month", "created_day", "created_hour", "cluster_"+str(n_cluster)
                   ]
    
    X_train, X_val,y_train, y_val = test_train(data_cluster, features)

    return(prediction(X_train,y_train, X_val, y_val))


# In[ ]:


compute_logloss(3, data)


# A tiny bit better but lets check with more clusters

# In[ ]:


log_loss_cls={}
for n in range(4,15):
    log_loss_cls[n]=compute_logloss(n, data)
    
n_c = sorted(log_loss_cls.items()) 
x, y = zip(*n_c) 
plt.plot(x, y)
plt.title("log_loss for different numbers of clusters")
plt.show()


# In[ ]:


log_loss_cls


# It seems that the more clusters, the better the log_loss becomes. A the extreme of this each point is his own cluster and we are back to the lat ,lon original data. 
# On the renthop website there is a feature calle Price Comparison that shows the difference between the price of the listing and the median of its neighborhood. Lets create this feature from our new neighborhoods and the price and see if it brings any improvement to the log loss 

# In[ ]:


data=cluster_latlon(100, data)

clusters_price_map=dict(data.groupby(by="cluster_100")["price"].median())
data["price_comparison"]=data['price']-data["cluster_100"].map(clusters_price_map)


# In[ ]:



features_2 = ["bathrooms", "bedrooms", "price", "latitude", 'longitude',
             "num_photos", "num_features", "num_description_words",                    
                    "created_month", "created_day", "created_hour"
                   ]
X_train, X_val,y_train, y_val=test_train(data, features_2)

prediction(X_train,y_train, X_val, y_val)


# In[ ]:


features_price_comp = ["bathrooms", "bedrooms", "price", "latitude", 'longitude',
             "num_photos", "num_features", "num_description_words",                    
                    "created_month", "created_day", "created_hour", "price_comparison"
                   ]
X_train, X_val,y_train, y_val=test_train(data, features_price_comp)

prediction(X_train,y_train, X_val, y_val)


# SO it seems that there might be a little value to cross theses newly created neighborhoods with the price to create a variable that adds information and improves (a tiny bit) the log loss score. ¯\\_(ツ)_/¯
