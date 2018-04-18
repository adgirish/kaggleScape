
# coding: utf-8

# # Customer Segments

# In this notebook I will try to find a possible customer segmenetation enabling to classify customers according the their different purchases. I hope this information will be useful for the next prediction task. 
# Since there are thousands of products in the dataset I will rely on aisles, which represent categories of products. Even with aisles features will be too much so I will use Principal Component Analysis to find new dimensions along which clustering will be easier. I will then try to find possible explanations for the identified clusters.

# ## First Exploration

# In[ ]:



import numpy as np 
import pandas as pd 

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


orders = pd.read_csv('../input/orders.csv')
orders.head()


# In[ ]:


prior = pd.read_csv('../input/order_products__prior.csv')
prior.head()


# In[ ]:


train = pd.read_csv('../input/order_products__train.csv')
train.head()


# This is my undrstanding of the dataset structur:
# * users are identified by user_id in the orders csv file. Each row of the orders csv fil represents an order made by a user. Order are identified by order_id;
# 
# * Each order of a user is characterized by an order_number which specifies when it has been made with respect to the others of the same user;
# 
# * each order consists of a set of product each characterized by an add_to_cart_order feature representing the sequence in which they have been added to the cart in that order;
# 
# * for each user we may have n-1 prior orders and 1 train order OR n-1 prior orders and 1 test order in which we have to state what products have been reordered.

# In[ ]:


##Due to the number of rows I have to reduce the set of prior data to publish the kernel 
##comment this if you execute it on your local machine
prior = prior[0:300000]


# In[ ]:


order_prior = pd.merge(prior,orders,on=['order_id','order_id'])
order_prior = order_prior.sort_values(by=['user_id','order_id'])
order_prior.head()


# In[ ]:


products = pd.read_csv('../input/products.csv')
products.head()


# In[ ]:


aisles = pd.read_csv('../input/aisles.csv')
aisles.head()


# In[ ]:


print(aisles.shape)


# In[ ]:


_mt = pd.merge(prior,products, on = ['product_id','product_id'])
_mt = pd.merge(_mt,orders,on=['order_id','order_id'])
mt = pd.merge(_mt,aisles,on=['aisle_id','aisle_id'])
mt.head(10)


# In[ ]:


mt['product_name'].value_counts()[0:10]


# In[ ]:


len(mt['product_name'].unique())


# In[ ]:


prior.shape


# ## Clustering Customers

# We are dealing with  143 types of product (aisle).

# In[ ]:


len(mt['aisle'].unique())


# Fresh fruits and fresh vegetables are the best selling goods.

# In[ ]:


mt['aisle'].value_counts()[0:10]


# I want to find a possible clusters among the different customers and substitute single user_id with the cluster to which they are assumed to belong. Hope this would eventually increase the next prediction model performance.
# 
# Ths first thing to do is creating a dataframe with all the purchases made by each user

# In[ ]:


cust_prod = pd.crosstab(mt['user_id'], mt['aisle'])
cust_prod.head(10)


# In[ ]:


cust_prod.shape


# We can then execute  a Principal Component Analysis to the obtained dataframe. This will reduce the number of features from the number of aisles to 6, the numbr of principal components I have chosen.

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(cust_prod)
pca_samples = pca.transform(cust_prod)


# In[ ]:


ps = pd.DataFrame(pca_samples)
ps.head()


# I haven plotted several pair of components looking for the one suitable, in my opinion,  for a KMeans Clustering.  I have chosen the (PC4,PC1) pair. Since each component is the projection of all the points of the original dataset I think each component is representative of the dataset. 

# In[ ]:


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
tocluster = pd.DataFrame(ps[[4,1]])
print (tocluster.shape)
print (tocluster.head())

fig = plt.figure(figsize=(8,8))
plt.plot(tocluster[4], tocluster[1], 'o', markersize=2, color='blue', alpha=0.5, label='class1')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

clusterer = KMeans(n_clusters=4,random_state=42).fit(tocluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)
print(centers)


# In[ ]:


print (c_preds[0:100])


# Here is how our clusters appear

# In[ ]:


import matplotlib
fig = plt.figure(figsize=(8,8))
colors = ['orange','blue','purple','green']
colored = [colors[k] for k in c_preds]
print (colored[0:10])
plt.scatter(tocluster[4],tocluster[1],  color = colored)
for ci,c in enumerate(centers):
    plt.plot(c[0], c[1], 'o', markersize=8, color='red', alpha=0.9, label=''+str(ci))

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()


# We have found a possible clustering for our customers. Let's check if we also manage to find some interesting pattern beneath it.

# In[ ]:


clust_prod = cust_prod.copy()
clust_prod['cluster'] = c_preds

clust_prod.head(10)


# In[ ]:


print (clust_prod.shape)
f,arr = plt.subplots(2,2,sharex=True,figsize=(15,15))

c1_count = len(clust_prod[clust_prod['cluster']==0])

c0 = clust_prod[clust_prod['cluster']==0].drop('cluster',axis=1).mean()
arr[0,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c0)
c1 = clust_prod[clust_prod['cluster']==1].drop('cluster',axis=1).mean()
arr[0,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c1)
c2 = clust_prod[clust_prod['cluster']==2].drop('cluster',axis=1).mean()
arr[1,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c2)
c3 = clust_prod[clust_prod['cluster']==3].drop('cluster',axis=1).mean()
arr[1,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c3)
plt.show()


# Let's check out what are the top 10 goods bought by people of each cluster. We are going to rely first on the absolute data and then on a percentage among the top 8 products for each cluster.

# In[ ]:


c0.sort_values(ascending=False)[0:10]


# In[ ]:


c1.sort_values(ascending=False)[0:10]


# In[ ]:


c2.sort_values(ascending=False)[0:10]


# In[ ]:


c3.sort_values(ascending=False)[0:10]


# A first analysis of the clusters confirm the initial hypothesis that:
# 
# * fresh fruits                     
# * fresh vegetables                 
# * packaged vegetables fruits       
# * yogurt                           
# * packaged cheese                   
# * milk                              
# * water seltzer sparkling water     
# * chips pretzels                    
# 
# are products which are genereically bought by the majority of the customers.
# 
# What we can inspect here is if clusters differ in quantities and proportions, with respect of these goods, or if a cluster is characterized by some goods not included in this list. For instance we can already see cluster 3 is characterized by 'Baby Food Formula' product which is a significant difference with respect to the other clusters.

# In[ ]:


from IPython.display import display, HTML
cluster_means = [[c0['fresh fruits'],c0['fresh vegetables'],c0['packaged vegetables fruits'], c0['yogurt'], c0['packaged cheese'], c0['milk'],c0['water seltzer sparkling water'],c0['chips pretzels']],
                 [c1['fresh fruits'],c1['fresh vegetables'],c1['packaged vegetables fruits'], c1['yogurt'], c1['packaged cheese'], c1['milk'],c1['water seltzer sparkling water'],c1['chips pretzels']],
                 [c2['fresh fruits'],c2['fresh vegetables'],c2['packaged vegetables fruits'], c2['yogurt'], c2['packaged cheese'], c2['milk'],c2['water seltzer sparkling water'],c2['chips pretzels']],
                 [c3['fresh fruits'],c3['fresh vegetables'],c3['packaged vegetables fruits'], c3['yogurt'], c3['packaged cheese'], c3['milk'],c3['water seltzer sparkling water'],c3['chips pretzels']]]
cluster_means = pd.DataFrame(cluster_means, columns = ['fresh fruits','fresh vegetables','packaged vegetables fruits','yogurt','packaged cheese','milk','water seltzer sparkling water','chips pretzels'])
HTML(cluster_means.to_html())


# The following table depicts the percentage these goods with respect to the other top 8 in each cluster. It is easy some interesting differences among the clusters. 
# 
# It seems people of cluster 1 buy more fresh vegetables than the other clusters. As shown by absolute data, Cluster 1 is also the cluster including those customers buying far more goods than any others.
# 
# People of cluster 2 buy more yogurt than people of the other clusters.
# 
# Absolute Data shows us People of cluster 3 buy a Lot of 'Baby Food Formula' which not even listed in the top 8 products but mainly characterize this cluster. Coherently (I think) with this observation they buy more milk than the others.

# In[ ]:


cluster_perc = cluster_means.iloc[:, :].apply(lambda x: (x / x.sum())*100,axis=1)
HTML(cluster_perc.to_html())


# I think another interesting information my come by lookig at the 10th to 15th most bought products for each cluster which will not include the generic products (i.e. vegetables, fruits, water, etc.) bought by anyone.

# In[ ]:


c0.sort_values(ascending=False)[10:15]


# In[ ]:


c1.sort_values(ascending=False)[10:15]


# In[ ]:


c2.sort_values(ascending=False)[10:15]


# In[ ]:


c3.sort_values(ascending=False)[10:15]


# As you can note by taking into account more products clusters start to differ significantly. I hope this informtion will be useful in the next prediction task.
