
# coding: utf-8

# ## Introduction
# 
# The following analysis is inspired by a notebook by omolluska ([When/ why are stocks bought and sold][1]). Check it out :). Here we build on that and try to gain additional insights about the id's (financial instruments) in the data and perform some clustering. The main idea is to look at the cumulative returns of each id and see how it correlates to other id's. That will allow us to cluster these in an easy way and see of th. 
# 
#  The last plot gives the main result (scroll to the end).
# 
# ##Caution! Important!
# 
# *The analysis changed somewhat to see if the result is not a statistical fluke. Let's assume we have lots of randomly generated curves (e.g. cumulative return) and we look at their correlations. Then we might find a subset of curves that are highly correlated and go up and about the same number of curves that are highly correlated and go down. The end result might look like what we got here. To see if the findings are for real the clustering is now performed up to the timestamp 900, ignoring everything that happens after that. The result is presented for all timestamps up to 1812. The new interpretation: **There are probably no clusters with respect to the cumulative returns** (at least not with the methodology used here) as the upward and downward trend of the clusters suddenly levels of after the timestamp 900 (in the old analysis these trends continued).*
# 
#   [1]: https://www.kaggle.com/sankhamukherjee/two-sigma-financial-modeling/when-why-are-stocks-bought-and-sold

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


# In[ ]:


data = pd.read_hdf("../input/train.h5")


# We will calculate correlations between all the id's but to ensure that we have enough overlap we only look at those id's that have more than 450 timestamps (i.e. at least 50% of data available).
# Additionally, we will later only keep id's that are generally highly correlated with other id's. That's what the parameter N_corr_cut is good for.

# In[ ]:


## Params

N_timestamps = 450 # how many timestamps that are non-nan for each id
N_corr_cut = 0.4 # min mean correlation coefficient when dropping id's


# In[ ]:


## data is used for clustering --> see comment at the beginning
data_full = data.copy()
data = data.iloc[0:800000,:]


# In[ ]:


## Check max timestamp
np.max(data.timestamp)


# In[ ]:


## Select only id's where sufficient data is available, 
## i.e. all timestamps available

select_ids = data[["id","y"]].groupby("id").count()

selected_ids = select_ids[select_ids.y > N_timestamps]#== N_timestamps]

selected_ids = np.array(selected_ids.index)

index_ids = [i in selected_ids for i in data.id]

data_corr = data[index_ids][["id","timestamp","y"]].copy()


# In[ ]:


index_ids = [i in selected_ids for i in data_full.id]
data_corr_full = data_full[index_ids][["id","timestamp","y"]].copy()


# ## Finding Clusters

# In[ ]:


## Create a dataframe where each id is a column

df_id = data_corr[['id', 'timestamp', 'y']].pivot_table(values='y',
                                     index='timestamp',columns='id')

df_id_full = data_corr_full[['id', 'timestamp', 'y']].pivot_table(values='y',
                                     index='timestamp',columns='id')


# In[ ]:


## Calculate the cumulative sum of the return y 
## and substract the mean of the cumulative sum at each timestamp

df_id_cumsum = df_id.cumsum()

diff = df_id_cumsum.mean(axis=1)
df_id_cumsum = df_id_cumsum.subtract(diff.values,axis="rows")

## Full Data

df_id_cumsum_full = df_id_full.cumsum()

diff = df_id_cumsum_full.mean(axis=1)
df_id_cumsum_full = df_id_cumsum_full.subtract(diff.values,axis="rows")

print(df_id_cumsum.shape)
df_id_cumsum.head()


# In[ ]:


## Check max timestamp
np.max(df_id_cumsum.index.values)


# We are left with 895 id's. Now let's look at the correlations between them and how these are distributed.

# In[ ]:


## Calculate the correlations between the id's

corr_cumsum = df_id_cumsum.corr()

dist = corr_cumsum.as_matrix()


# In[ ]:


plt.hist(dist.flatten(),bins=100)
plt.title("Distribution of Correlations Between Id's");


# The bimodal distribution vanished after changing the analysis (see comment at beginning of notebook).

# In[ ]:


## Look at id's that are generally strongly correlated to others

dist_id_mean = np.mean(np.abs(dist),axis = 1)
index_mean = dist_id_mean > N_corr_cut

tmp_cut = dist[index_mean,:]
tmp_cut = tmp_cut[:,index_mean]

print("Number of id's %i" % (tmp_cut.shape[0]))

plt.hist(tmp_cut.flatten(),bins=100)
plt.title("Distribution of Correlations Between Id's");


# Now, we would like to see if the id's form any clusters.

# In[ ]:


g = sns.clustermap(tmp_cut,metric="euclidean",method="average")


# Indeed, there are two clusters. Within each cluster the correlations are high and positive. In between clusters these correlations are inverted. Next, we extract these clusters and analyze them.

# In[ ]:


## Perform Kmeans to easily get the two clusters

clf = KMeans(n_clusters = 2)
clf.fit(tmp_cut)
labels = clf.labels_

print("%i id's in cluster 1 and %i id's in cluster 2" % (np.sum(labels == 0),np.sum(labels == 1)))


# In[ ]:


## See if Kmeans got the clusters right

g = sns.clustermap(tmp_cut[labels == 0,:],metric="euclidean",method="average",square=True)


# Looks good. Finally, we will look at the mean cumulative returns for each cluster and compare them. Note that we have already subtracted the general upward trend of the cumulative returns. For comparison, we include the result without that adjustment.

# ## Mean Cumulative Returns for Each Cluster

# In[ ]:


## Define one dataframe for each cluster

ids = corr_cumsum.columns[index_mean]

ids_1 = ids[labels == 0]
ids_2 = ids[labels == 1]

data_sub_c1 = df_id_cumsum_full[ids_1]
data_sub_c2 = df_id_cumsum_full[ids_2]

## Cumulative Sums Without Adjustments

df_id_cumsum_no_adjust = df_id_full.cumsum()

data_sub_c1_no_adjust = df_id_cumsum_no_adjust[ids_1]
data_sub_c2_no_adjust = df_id_cumsum_no_adjust[ids_2]


# In[ ]:


## Determine Mean of Cumulative Returns

## Without Adjustment

returns_1 = data_sub_c1_no_adjust.mean(axis=1)
returns_2 = data_sub_c2_no_adjust.mean(axis=1)

std_1 = data_sub_c1_no_adjust.std(axis=1)
std_2 = data_sub_c2_no_adjust.std(axis=1)

plt.plot(returns_1,alpha=1)
plt.plot(returns_2,alpha=1)
plt.title("Mean Cumulative Returns for Each Cluster, without adjustment")
plt.xlabel("Timestamp")
plt.ylabel("Mean Cumulative Return");

plt.fill_between(returns_1.index, returns_1 - std_1, returns_1 + std_1, color='b', alpha=0.05)
plt.fill_between(returns_2.index, returns_2 - std_2, returns_2 + std_2, color='g', alpha=0.05)

plt.show()

## With Adjustment

returns_1 = data_sub_c1.mean(axis=1)
returns_2 = data_sub_c2.mean(axis=1)

std_1 = data_sub_c1.std(axis=1)
std_2 = data_sub_c2.std(axis=1)

plt.plot(returns_1,alpha=1)
plt.plot(returns_2,alpha=1)
plt.title("Mean Cumulative Returns for Each Cluster, with adjustment")
plt.xlabel("Timestamp")
plt.ylabel("Mean Cumulative Return");

plt.fill_between(returns_1.index, returns_1 - std_1, returns_1 + std_1, color='b', alpha=0.05)
plt.fill_between(returns_2.index, returns_2 - std_2, returns_2 + std_2, color='g', alpha=0.05);


# ## Caution
# 
# *As mentioned in the beginning the clustering is now only performed up to the timestamp 900. This changed the end result quite a bit. Now we can see that the mean cumulative return levels of after 900 and the standard deviation starts to increase. This should be expected for curves that behave in a random way. The fact that they mirror each other closely is probably due to adjusting for the general upward trend in the cumulative mean.*
# 
# The two curves show the mean cumulative return within each cluster over time. The shaded regions show the standard deviation of the cumulative return within each cluster.
