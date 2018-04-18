
# coding: utf-8

# As shown before, some features per id can be clustered intro groups that perform the same kind of time evolution.  Unfortunately a unique feature does not perform the same dynamics for different ids. But maybe there are groups of ids that have identical or similar features and perhaps this groups can give some insights into the "global" dynamics. 
# 
# ## Assumptions: ##
# 
#  - NaN values make sense. An id that has always NaNs for a specific feature has no relationship with it.  
#  - Id's live in different time-zones (and have different lifetimes)

# In[ ]:


import numpy as np
import pandas as pd
import kagglegym
import matplotlib.pyplot as plt

with pd.HDFStore("../input/train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")

# Create an environment
env = kagglegym.make()

# Get first observation
observation = env.reset()

# Get the train dataframe
train = observation.train


# **Lifetimes** 
# 
# First, I will pick up the idea of Chase, to [look at id lifetimes][1]. There are a lot of ids having a large number of timestamps and maybe they could give more insights into the time evolution and dynamics of features. Perhaps one could also find id-groups of same lifetimes with similar behavior. 
# 
#   [1]: https://www.kaggle.com/chaseos/two-sigma-financial-modeling/understanding-id-and-timestamp

# In[ ]:


lifetimes = df.groupby('id').apply(len)
lifetimes = lifetimes.sort_values(ascending=False)
lifetimes = lifetimes.reset_index()
lifetimes.columns = ["id", "duration"]
lifetimes.head()


# Let's collect all id's that have a lifetime of 1813.

# In[ ]:


long_lifetime_ids = lifetimes[lifetimes["duration"] == 1813]
long_lifetime_ids.info()


# **Select ids of same nan-structure**
# 
# There are 527 ids with a lifetime of 1813 timestamps. As I want to study the feature dynamics of id's that behave the same way, I need to find all those id's that share the same features (and nan-structures). Let's do this by a simple approach: Select one example id of the dataframe "long_lifetimes_ids" and find all ids in that frame that match its feature-presence (nan-structure). 
# 
# I will be careful with binary transformations because a nan-value may not be present all over the time of my selected id and perhaps 0.0 is the first value that occurs in the situation where "nan" changes to a value.  Instead, I will do the following:
# 
#  - Kick out features that have a permanent nan-structure over all 1813 timestamps
#  - Keep features with partial present nan-structures, but collect their labels for safety 

# In[ ]:


long_lifetime_ids.head()


# In[ ]:


def find_nan_structure(instrument, data):
    data_id = data.loc[data["id"]==instrument,:]
    no_nan_features = []
    partial_nan_features = []
    total_nan_features = [] 
    for col in data_id.columns:
        if col not in ["id", "timestamp", "y"]:
            nr_nans = pd.isnull(data_id[col]).sum()
            if (nr_nans == 0):
                no_nan_features.append(col)
            elif (nr_nans == len(data_id[col])):
                total_nan_features.append(col)
            else:
                partial_nan_features.append(col)
    return no_nan_features, total_nan_features, partial_nan_features
    


# By playing around, I found out that id 711 belongs to a large group of ids that share the same features. 

# In[ ]:


no_nan, total_nan, partial_nan = find_nan_structure(711, df)


# In[ ]:


def find_id_group(instrument, data, lifetime_ids):
    strong_cluster = []
    soft_cluster = []
    no_nan, total_nan, partial_nan = find_nan_structure(instrument, data)
    no_nan_soft = set(no_nan).union(partial_nan)
    for element_id in lifetime_ids:
        no_nan_e, total_nan_e, partial_nan_e = find_nan_structure(element_id, data)
        no_nan_soft_e = set(no_nan_e).union(partial_nan_e)
        if set(no_nan_soft_e) == set(no_nan_soft):
            soft_cluster.append(element_id)
        if set(no_nan_e) == set(no_nan):
            strong_cluster.append(element_id)
    return strong_cluster, soft_cluster


# In[ ]:


strong_cluster, soft_cluster = find_id_group(711, df, long_lifetime_ids.id)


# In[ ]:


strong_cluster


# I will proceed with the strong-cluster group of id 711. Let's create a dataframe which contains only these ids and let's try to find correlated features or do some other stuff. ;-) 

# In[ ]:


cluster_data = df[df["id"].isin(strong_cluster)]
cluster_data.head()


# Let's select a feature, which does not contain nan-values, for playing around: 

# In[ ]:


test_feature = no_nan[1]
plt.figure()
for instrument in strong_cluster:
    plt.plot(cluster_data[cluster_data["id"]==instrument].timestamp, cluster_data[cluster_data["id"]==instrument][test_feature].values, '.-')
plt.xlabel("timestamp")
plt.ylabel(test_feature)


# In[ ]:


def find_id_groups(data, idlist, feature, limit):
    groups = []
    singles = []
    for list_instrument in idlist:
        group = []
        for next_instrument in idlist:
            coeff = np.corrcoef(data.ix[data.id==list_instrument, feature].values, data.ix[data.id==next_instrument, feature].values)[0,1]
            coeff = np.round(coeff, decimals=2)
            if coeff >= limit:
                group.append(next_instrument)
        for member in group:
            while member in idlist:
                idlist.remove(member)
        if len(group) > 1:
            groups.append(group)
        elif len(group) == 1:
            singles.append(list_instrument)
    return groups, singles


# In[ ]:


id_list = strong_cluster[:]
groups, singles = find_id_groups(cluster_data, id_list, test_feature, 0.80)
groups


# Yeah! Found a pattern again! :-) Just by looking at "stupid" linear correlations, we can find that different features of one id are correlated but also that values of a specific feature of different ids are correlated in some cases. Let's have a look at those groups:  

# In[ ]:


for group in groups:
    plt.figure()
    for instrument in group:
        plt.plot(cluster_data[cluster_data.id==instrument]["timestamp"].values, cluster_data[cluster_data.id==instrument][test_feature].values, '.-')
    plt.xlabel("timestamp")
    plt.ylabel(test_feature)


# :-D
