
# coding: utf-8

# **Cluster Visualization - Assets based on Structural NAN values**

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
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.


# In[ ]:


#load the data. It comes in train.h5 so we need to us HDFStore
df = pd.HDFStore("../input/train.h5", "r").get("train")


# In[ ]:


# I make vectors that hold the ratio of NAN values for a given id in a given column
unique_ids = pd.unique(df.id)
len(unique_ids)
NaN_vectors = np.zeros(shape=(len(unique_ids), df.shape[1]))

for i, i_id in enumerate(unique_ids):
    data_sub = df[df.id ==i_id]
    NaN_vectors[i,:] = np.sum(data_sub.isnull(),axis=0) /float(data_sub.shape[0])
    
NaN_vectors


# In[ ]:


# get all the NaN vectors in which every collumn for that ID is NaN. What we are looking for 
#is collumns in which the features fundamentally do not exist. 
bin_NaN = 1*(NaN_vectors==1)
print("Still has the shape of {} by {}".format(bin_NaN.shape[0],bin_NaN.shape[1]))


# In[ ]:


# we now have a vector of things that are either 1 where nothing exists in teh column or zero something
#exists Now we take a covariance over these bins to see which ones move togehter. we are looking only based on columns
bin_cov=np.corrcoef(bin_NaN.T)
bin_cov.shape[1]


# In[ ]:


# plot bin_cov
plt.matshow(bin_cov)

#if you think abou what this shows it is show the probability that when an entire column is missing what
# is the probability that another column will be completely missing. 


# In[ ]:


# In this graph i make the matrix sparse by considering only things that have perfect correlation. This
# gives us insight into the relationship of the pairs.
plt.matshow(bin_cov == 1) 


# In[ ]:


# What we are doing here is looking at all the column pairs that when they are missing  they are always
# missing together. ie they have a corralation of 1. We also get the count. it stands to reason that
# if this happens in only one or two id's out of 1400 then perhaps it is a statistical anomoly or could be 
# reflective of a non structural issue. This is actually very enlightening and we see there are 60
# some odd pairs that satisfy this criteria. More importantly is that it happens for lots of tickers.
# Maybe we have soemthing here.
bin_NaN
edges = []
count =np.dot(bin_NaN.T,bin_NaN)
for i in range(bin_cov.shape[0]):
    for j in range(bin_cov.shape[0]-i):
        if i!=i+j and bin_cov[i,i+j]==1:
            edges.append([i,i+j,count[i,i+j]])
print(edges)


# In[ ]:


#lets see how many unique counts there are. it looks like a few of these counts happen multiple times.
# this is interesting and could imply some structural issue.
ucount = [i[2] for i in edges]
print(np.unique(ucount))


# In[ ]:


print('rows: {}'.format(bin_NaN.shape[0]))
print('cols: {}'.format(len(edges)))

# the idea here is that we create a feature vector. We look at all the ids which have all their data
# missing in a certain collumn. above we found that if all the data is missing in a certain collumn it
# would be missing in another collumn as well. so we look at all these pairs (shown as edges) and we 
# then create a matrix of id x edges. We then put a 0 or a 1 in the collumn to indicate that the pair of 
# data is missing or not. This serves as a feature. I will then go on to cluster over these features. 

nan_features = np.zeros((bin_NaN.shape[0],len(edges)))
for i in range(bin_NaN.shape[0]):
    for j, edge in enumerate(edges):
        nan_features[i,j] = 1*(bin_NaN[i,edge[0]] & bin_NaN[i,edge[1]])

print('this is just a check that indexing is correct: {}'.format(np.sum(nan_features,axis=1).shape[0]))


# In[ ]:


# we take a look at the silouette score as we increase the number of clusters to understand the optimal
#number of clusters. We see here that it continues to increase which we would expect. I chose to cut it
# off around 12
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Range for k
kmin = 2
kmax = 25
sil_scores = []

#Compute silouhette scoeres
for k in range(kmin,kmax):
    km = KMeans(n_clusters=k, n_init=20).fit(nan_features)
    sil_scores.append(silhouette_score(nan_features, km.labels_))

#Plot
plt.plot(range(kmin,kmax), sil_scores)
plt.title('KMeans Results')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# In[ ]:


# DBSCAN The only thing that is important from here is the labels. 
# the way that DB scan works is that you give it eps and min_samples and it
# finds core groups. eps is the distance cut off and min is how many elements
# at minimum you need to define a cluster

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

X = nan_features

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)


# In[ ]:


#COLOR MAPPING - we run a kmeans cluster but we need to decide how many
#clusters we want to use. This is another way for us to cluster. Here we use 
k=12
km = KMeans(n_clusters=k, n_init=20).fit(nan_features)
colors=km.labels_


# In[ ]:


#now we try to visualize the data of these features.
# WOW LOOK AT THESE RESULTS. THAT IS BEAUTIFUL!!

from sklearn.manifold import TSNE
from time import time

n_iter = 5000

for i in [2, 5, 30, 50, 100]:
    t0 = time()
    model = TSNE(n_components=2, n_iter = n_iter,random_state=0, perplexity =i)
    np.set_printoptions(suppress=True)
    Y = model.fit_transform(nan_features)
    t1 =time()

    print( "t-SNE: %.2g sec" % (t1 -t0))
    plt.scatter(Y[:, 0], Y[:, 1], c= colors)
    plt.title('t-SNE with perplexity = {}'.format(i))
    plt.show()


# In[ ]:


# Now i do the same in 3d to try to better understand these clusters
from mpl_toolkits.mplot3d import Axes3D


n_iter = 5000

for i in [2, 5, 30, 50, 100]:
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    t0 = time()
    model = TSNE(n_components=3, random_state=0, perplexity=i, n_iter=n_iter)
    np.set_printoptions(suppress=True)

    Y = model.fit_transform(nan_features)
    t1 =time()

    print( "t-SNE: %.2g sec" % (t1 -t0))

    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2],c=db.labels_,
               cmap=plt.cm.Paired)
    ax.set_title("3D T-SNE - Perplexity = {}".format(i))
    ax.set_xlabel("1st dim")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd dim")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd dim")
    ax.w_zaxis.set_ticklabels([])
    plt.show()


# In[ ]:


# I will use PCA now to plot
from sklearn import decomposition

# I chose this number pretty much at random, you can change it. using 22 features to describe
# something with 110 x variables still seems high.
n_eigens = 22
# Creating PCA object
pca = decomposition.PCA(n_components=n_eigens, svd_solver ='randomized', whiten=True)
X_pca =pca.fit_transform(nan_features)
X_pca


# **2D -PCA Plot**

# In[ ]:


# This is a 2D pca plot... mehhhh
plt.scatter(X_pca[:,0],X_pca[:,1],c=colors)


# In[ ]:


# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = decomposition.PCA(n_components=3).fit_transform(nan_features)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],c=colors,
           cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[ ]:


# here we plot a graph that looks at how many PCA components explain the variation
n_eigens=10
X_reduced = decomposition.PCA(n_components=n_eigens).fit(nan_features)
with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(8, 5));
    plt.title('Explained Variance Ratio over Component');
    plt.plot(X_reduced.explained_variance_ratio_);


# In[ ]:


with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(8, 5));
    plt.title('Cumulative Explained Variance over EigenFace');
    plt.plot(X_reduced.explained_variance_ratio_.cumsum());


# In[ ]:


print('PCA captures {:.2f} percent of the variance in the dataset'.format(X_reduced.explained_variance_ratio_.sum() * 100))
print('PCA components have dimensions {} by {}'.format(*X_reduced.components_.shape))

