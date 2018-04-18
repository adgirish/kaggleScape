
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# 
# Gene analysis is something which is widely used in biological science.Many of the Data learners coming here will be haveing trouble with these weird transciption numbers. So why not just conver this into just two columns and then apply our analysis.

# In[ ]:


df= pd.read_csv('../input/data_set_ALL_AML_train.csv')
df.head()


# In[ ]:


df1 = [col for col in df.columns if "call" not in col]
df = df[df1]
df.head()


# In[ ]:


df.T.head()


# In[ ]:


df = df.T
df2 = df.drop(['Gene Description','Gene Accession Number'],axis=0)
df2.index = pd.to_numeric(df2.index)
df2.sort_index(inplace=True)
df2.head()


# Now see here we are having just 38 columns which are our respective people

# In[ ]:


df2['cat'] = list(pd.read_csv('../input/actual.csv')[:38]['cancer'])
dic = {'ALL':0,'AML':1}
df2.replace(dic,inplace=True)
df2.head(3)


# # PCA analysis
# #### Principal Component Analysis (or PCA) uses linear algebra to transform the dataset into a compressed form.
# 
# #### Generally this is called a data reduction technique. A property of PCA is that you can choose the number of dimensions or principal component in the transformed result.
# 
# #### In this case we will use it to analyse the feature importanace
# 
# We will reduce the data into two features.

# ![](https://media.giphy.com/media/QA44U3zAnH0ha/source.gif)

# In[ ]:


from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(df2.drop('cat',axis=1))

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=30)
Y_sklearn = sklearn_pca.fit_transform(X_std)


# In[ ]:


cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()

sklearn_pca.explained_variance_ratio_[:10].sum()

cum_sum = cum_sum*100

fig, ax = plt.subplots(figsize=(8,8))
plt.bar(range(30), cum_sum, label='Cumulative _Sum_of_Explained _Varaince', color = 'b',alpha=0.5)
plt.title("Around 95% of variance is explained by the Fisrt 30 colmns ");


# In[ ]:


X_reduced2 = Y_sklearn


# In[ ]:


df2.cat.values


# In[ ]:


train = pd.DataFrame(X_reduced2)
train['cat'] =  df2['cat'].reset_index().cat
train.head(3)


# ## To just give the glimpse of what PCA does to data, I am reducing it to 3 components,But in final modeling we will use 30 var.**

# In[ ]:


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=3)
X_reduced  = sklearn_pca.fit_transform(X_std)
Y=train['cat']
from mpl_toolkits.mplot3d import Axes3D
plt.clf()
fig = plt.figure(1, figsize=(10,6 ))
ax = Axes3D(fig, elev=-150, azim=110,)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired,linewidths=10)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(1, figsize=(10,6))
plt.scatter(X_reduced[:, 0],  X_reduced[:, 1], c=df2['cat'],cmap=plt.cm.Paired,linewidths=10)
plt.annotate('See The Brown Cluster',xy=(20,-20),xytext=(9,8),arrowprops=dict(facecolor='black', shrink=0.05))
#plt.scatter(test_reduced[:, 0],  test_reduced[:, 1],c='r')
plt.title("This The 2D Transformation of above graph ")


# ### Wow , See how we got our clusters and we differentiated cancer disease from it. We can think of this as a cluster.
# Here PCA play a very important role.
# 

# In[ ]:


test = pd.read_csv('../input/data_set_ALL_AML_independent.csv')

test.head(3)


# In[ ]:


test1 = [col for col in test.columns if "call" not in col]
test = test[test1]
test = test.T
test2 = test.drop(['Gene Description','Gene Accession Number'],axis=0)
test2.index = pd.to_numeric(test2.index)
test2.sort_index(inplace=True)
#test2['cat'] = list(pd.read_csv('actual.csv')[39:63]['cancer'])
#dic = {'ALL':0,'AML':1}
#test2.replace(dic,inplace=True)
#test2


# In[ ]:


from sklearn.preprocessing import StandardScaler
Y_std = StandardScaler().fit_transform(test2)

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=30)
test_reduced = sklearn_pca.fit_transform(Y_std)


# In[ ]:


test_set = pd.DataFrame(test_reduced)

test_set.head(3)


# ### As clear from graph it is useful to use KNN to find the required weather that it is ALL or AML.
# Depending upon distance

# In[ ]:


train.drop('cat',axis=1).plot(kind='hist',figsize=(8,10))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf= KNeighborsClassifier(n_neighbors=10,)
clf.fit(train.drop('cat',axis=1),train['cat'])


# In[ ]:


pred = clf.predict(test_set)

pateint = pd.read_csv('../input/actual.csv')['cancer'][38:]

true = pateint.replace(dic)

import sklearn
sklearn.metrics.confusion_matrix(true, pred)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=2)
clf.fit(train.drop('cat',axis=1),train['cat'])
pred = clf.predict(test_set)
true = pateint.replace(dic)
print(sklearn.metrics.confusion_matrix(true, pred))
print()


# In[ ]:


from sklearn import svm

clf=svm.SVC(kernel='linear')
clf.fit(train.drop('cat',axis=1),train['cat'])
pred = clf.predict(test_set)

pateint = pd.read_csv('../input/actual.csv')['cancer'][38:]

true = pateint.replace(dic)

print(sklearn.metrics.confusion_matrix(true, pred))
print()


# # Here the Red Points(means test set) is plotted on the training set to just give the illusion that which one false in which category

# In[ ]:


fig = plt.figure(1, figsize=(14,6))
plt.scatter(X_reduced[:, 0],  X_reduced[:, 1], c=df2['cat'],cmap=plt.cm.Paired,alpha=0.7,linewidths=7)
plt.scatter(test_reduced[:, 0],  test_reduced[:, 1],c='r',linewidths=10)


# **Inference:**<br>
# Above Analysis give the picture of how to differntiate the data.<br>
# **Note-** Accuracy is less due to insufficient data for training. Still if any measures to improve this model, Please suggest!.
# <br>
# ## Upvote!
# If you like my kernel than do upvote to boost my confidence :)
