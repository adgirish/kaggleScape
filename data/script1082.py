
# coding: utf-8

# ## Objective 
# Learn various types of clustering algorithms as available in sklearn, We will use "World Happiness Report data" as dataset for clustering algorithms.
# 
# **List of Clustering methods **
# 
# **1. K-Means
# 2. Mean Shift
# 3. Mini Batch K-Means
# 4. Spectral Clustering 
# 5. DBSCAN
# 6. Affinity Propagation
# 7. Birch
# 8. Gaussian Mixture Modeling**
# 
# ### Import Libraries
# 

# In[17]:


import numpy as np                   # Data manipulation
import pandas as pd                  # DataFrame manipulation
import time                          # To time processes 
import warnings                      # To suppress warnings
import matplotlib.pyplot as plt      # For Graphics
import seaborn as sns
from sklearn import cluster, mixture # For clustering 
from sklearn.preprocessing import StandardScaler

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ### Read Data From File 

# In[18]:


#os.chdir("E:/Big_Data/Code/Python/exercise/Clustering_Algorithms")
WHR_data = pd.read_csv("../input/2017.csv", header = 0)

#To play with "World Happiness Report" Data set, we will create a copy 
WHR_data_copy = WHR_data.copy(deep = True)

# preview Data
print(WHR_data.info())
WHR_data.shape
WHR_data.head()
WHR_data.sample(15)



# In[19]:


WHR_data.columns
plt.figure(figsize=(12,8))
sns.heatmap(WHR_data.corr())


# In[ ]:



sns.pairplot(WHR_data[['Country', 'Happiness.Rank', 'Happiness.Score', 'Whisker.high','Whisker.low', 'Economy..GDP.per.Capita.', 'Family']]);


# In[ ]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# For offline use
import cufflinks as cf

cf.go_offline()
WHR_data[['Happiness.Score','Whisker.high','Family','Freedom','Dystopia.Residual']].iplot(kind='spread')


# ### Explore & Scale

# In[20]:




#Ignore Country and Happiness_Rank Columns

WHR_data = WHR_data.iloc[:,2:]

print("\n \n Dimenstion of dataset  : WHR_data.shape")
WHR_data.shape

WHR_data.dtypes


# ### Normalize Dataset
# Normalize dataset for easier parameter selection, 
# 
# Creating Object of StandardScaler with default parameters
#  :- class sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
# 
# fit_transform(X[, y])	Fit to data, then transform it. 
# fit_transform to X and y with optional parameters fit_params and returns a transformed version of X.
# 

# In[21]:


# Instantiate Scaler Object 
ss = StandardScaler()
# Fit and transform 
ss.fit_transform(WHR_data)
WHR_data.sample(20)


# ## Begin Clustering 
# 
# We are performing clustering of unlabeled data thorugh sklearn.cluster module .
# 
# List of Clustering methods 
# 
# 1. K-Means
# 2. Mean Shift
# 3. Mini Batch K-Means
# 4. Spectral Clustering 
# 5. DBSCAN
# 6. Affinity Propagation
# 7. Birch
# 8. Gaussian Mixture Modeling 
# 
# 
# Each Clustering algorithm comes in two variants, A class that implements the fit method to learn the clusters on train data, and a function , that given the train data, returns an arrays of integer labels corresponding to the different clusters. For the class, the labels over the training data can be found in the labels_attribute

# ##### K-Means
# The k-means algorithm divides a set of N samples X into K disjoint clusters C, each described by the mean \mu_j of the samples in the cluster. The means are commonly called the cluster “centroids”; note that they are not, in general, points from X, although they live in the same space. The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum of squared criterion:
# 
# ![alt_text](http://scikit-learn.org/stable/_images/math/1886f2c69775746ac7b6c1cdd88c53c676839015.png)

# ##### Mean Shift
# MeanShift clustering aims to discover blobs in a smooth density of samples. It is a centroid based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids.
# 
# Given a candidate centroid x_i for iteration t, the candidate is updated according to the following equation:
# 
# ![alt_text](http://scikit-learn.org/stable/_images/math/df67cad6c90923bd6d5dd1ba1cc98b73ba772bd8.png)
# 
# Where N(x_i) is the neighborhood of samples within a given distance around x_i and m is the mean shift vector that is computed for each centroid that points towards a region of the maximum increase in the density of points. This is computed using the following equation, effectively updating a centroid to be the mean of the samples within its neighborhood:
# 
# ![alt_text](http://scikit-learn.org/stable/_images/math/64f9e995cea2b11641d37f2ec1cfcf1d590d2797.png)
# 

# ##### Mini Batch K-Means
# 
# The MiniBatchKMeans is a variant of the KMeans algorithm which uses mini-batches to reduce the computation time, while still attempting to optimise the same objective function. Mini-batches are subsets of the input data, randomly sampled in each training iteration. These mini-batches drastically reduce the amount of computation required to converge to a local solution. In contrast to other algorithms that reduce the convergence time of k-means, mini-batch k-means produces results that are generally only slightly worse than the standard algorithm.
# 
# MiniBatchKMeans converges faster than KMeans, but the quality of the results is reduced. In practice this difference in quality can be quite small

# 
# ##### Spectral Clustering
# 
# SpectralClustering does a low-dimension embedding of the affinity matrix between samples, followed by a KMeans in the low dimensional space. It is especially efficient if the affinity matrix is sparse and the pyamg module is installed. SpectralClustering requires the number of clusters to be specified. It works well for a small number of clusters but is not advised when using many clusters.

# ##### DBSCAN
# The DBSCAN algorithm views clusters as areas of high density separated by areas of low density. Due to this rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped

# ##### Affinity Propagation
# AffinityPropagation creates clusters by sending messages between pairs of samples until convergence. A dataset is then described using a small number of exemplars, which are identified as those most representative of other samples. The messages sent between pairs represent the suitability for one sample to be the exemplar of the other, which is updated in response to the values from other pairs. This updating happens iteratively until convergence, at which point the final exemplars are chosen, and hence the final clustering is given.

# ##### Birch
# The Birch builds a tree called the Characteristic Feature Tree (CFT) for the given data. The data is essentially lossy compressed to a set of Characteristic Feature nodes (CF Nodes). The CF Nodes have a number of subclusters called Characteristic Feature subclusters (CF Subclusters) and these CF Subclusters located in the non-terminal CF Nodes can have CF Nodes as children.

# ##### Gaussian Mixture Modeling
# 
# A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.
# 

# 
# ### Define Cluster Class
# 

# In[ ]:



# Define CluserMethod class : which returns the clustering result based on input 

class ClusterMethodList(object) :
    def get_cluster_instance(self, argument,input_data,X):
        method_name = str(argument).lower()+ '_cluster'
        method = getattr(self,method_name,lambda : "Invalid Clustering method")
        return method(input_data,X)
    
    def kmeans_cluster(self,input_data,X):
        km = cluster.KMeans(n_clusters =input_data['n_clusters'] )
        return km.fit_predict(X)
   
    def meanshift_cluster(self,input_data,X):
        ms = cluster.MeanShift(bandwidth=input_data['bandwidth'])
        return  ms.fit_predict(X)
    
    def minibatchkmeans_cluster(self,input_data,X):
        two_means = cluster.MiniBatchKMeans(n_clusters=input_data['n_clusters'])
        return two_means.fit_predict(X)
   
    def dbscan_cluster(self,input_data,X):
        db = cluster.DBSCAN(eps=input_data['eps'])
        return db.fit_predict(X)
    
    def spectral_cluster(self,input_data,X):
        sp = cluster.SpectralClustering(n_clusters=input_data['n_clusters'])
        return sp.fit_predict(X)
   
    def affinitypropagation_cluster(self,input_data,X):
        affinity_propagation =  cluster.AffinityPropagation(damping=input_data['damping'], preference=input_data['preference'])
        affinity_propagation.fit(X)
        return affinity_propagation.predict(X)
       
    
    def birch_cluster(self,input_data,X):
        birch = cluster.Birch(n_clusters=input_data['n_clusters'])
        return birch.fit_predict(X)
   
    def gaussian_mixture_cluster(self,input_data,X):
        gmm = mixture.GaussianMixture( n_components=input_data['n_clusters'], covariance_type='full')
        gmm.fit(X)
        return  gmm.predict(X)



# In[ ]:


# Define Clustering Prcoess

def startClusteringProcess(list_cluster_method,input_data,no_columns,data_set):
    fig,ax = plt.subplots(no_rows,no_columns, figsize=(10,10)) 
    cluster_list = ClusterMethodList()
    i = 0
    j=0
    for cl in list_cluster_method :
        cluster_result = cluster_list.get_cluster_instance(cl,input_data,data_set)
        #convert cluster result array to DataFrame
        data_set[cl] = pd.DataFrame(cluster_result)
        ax[i,j].scatter(data_set.iloc[:, 4], data_set.iloc[:, 5],  c=cluster_result)
        ax[i,j].set_title(cl+" Cluster Result")
        j=j+1
        if( j % no_columns == 0) :
            j= 0
            i=i+1
    plt.subplots_adjust(bottom=-0.5, top=1.5)
    plt.show()



# #### Initialize Input Parameters for Clustering 

# In[ ]:


list_cluster_method = ['KMeans',"MeanShift","MiniBatchKmeans","DBScan","Spectral","AffinityPropagation","Birch","Gaussian_Mixture"]
# For Graph display 
no_columns = 2
no_rows = 4
# NOT all algorithms require this parameter
n_clusters= 3
bandwidth = 0.1 
# eps for DBSCAN
eps = 0.3
## Damping and perference for Affinity Propagation clustering method
damping = 0.9
preference = -200
input_data = {'n_clusters' :  n_clusters, 'eps' : eps,'bandwidth' : bandwidth, 'damping' : damping, 'preference' : preference}


# #### Plot Graph

# In[ ]:


# Start Clustering Process
startClusteringProcess(list_cluster_method,input_data,no_columns,WHR_data)


# In[ ]:


WHR_data.insert(0,'Country',WHR_data_copy.iloc[:,0])


# In[ ]:


WHR_data.iloc[:,[0,11,12,13,14,15,16,17,18]]


# ### Gaussian Mixture Clustering Visualization

# In[ ]:




data = dict(type = 'choropleth', 
           locations = WHR_data['Country'],
           locationmode = 'country names',
           z = WHR_data['Gaussian_Mixture'], 
           text = WHR_data['Country'],
           colorbar = {'title':'Cluster Group'})
layout = dict(title = 'Gaussian Mixture Clustering Visualization', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)



# ### K-Means Clustering Visualization

# In[ ]:


data = dict(type = 'choropleth', 
           locations = WHR_data['Country'],
           locationmode = 'country names',
           z = WHR_data['KMeans'], 
           text = WHR_data['Country'],
           colorbar = {'title':'Cluster Group'})
layout = dict(title = 'K-Means Clustering Visualization', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)


# ### Global Happiness Score Map

# In[ ]:



data = dict(type = 'choropleth', 
           locations = WHR_data['Country'],
           locationmode = 'country names',
           z = WHR_data['Happiness.Score'], 
           text = WHR_data['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Global Happiness Score', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)

