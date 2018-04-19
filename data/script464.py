
# coding: utf-8

# As there are only 4200-ish samples and 350-ish features, I strongly believe this can hit the Curse of Dimensionality. I therefore think a dimensionality reduction is good for this dataset. Let's investigate some methods.

# In[ ]:


# Import the important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


# Read the data
df = pd.read_csv("../input/train.csv").set_index("ID")


# There are some features that are constants. Let's identify them by looking at the standard deviation (check id std ==0.0) and then drop those features.

# In[ ]:


desc = df.describe().transpose()
columns_to_drop = desc.loc[desc["std"]==0].index.values
df.drop(columns_to_drop, axis=1, inplace=True)


# Just to check which columns we just dropped:

# In[ ]:


print(columns_to_drop)


# There are some categorical features in X0-X8. Let's count the cardinality and label encode those.

# In[ ]:


df08 = df[["X{}".format(x) for x in range(9) if x != 7]]


# In[ ]:


tot_cardinality = 0
for c in df08.columns.values:
    cardinality = len(df08[c].unique())
    print(c, cardinality)
    tot_cardinality += cardinality
print(tot_cardinality)


# We can do some guesses what these are. Can X3 be the day of week? Can X6 be the months of year?  Let's do the label encoding:
# **Update**: Label encoding does not make sense. I'm updating this to One-Hot encoding.

# In[ ]:


df = pd.get_dummies(df, columns=["X{}".format(x) for x in range(9) if x != 7])


# I've heard there is an outlier in the target variable. Let's look and remove it.

# In[ ]:


# sns.distplot(df.y)
#(Why do I get a warning?)
# I get a long warning on the kaggle kernel, I'm commenting this line.


# In[ ]:


# Drop it!
df.drop(df.loc[df["y"] > 250].index, inplace=True)


# ## PCA - Principal component analysis.
# For the sake of simplicity, do a 2-dimensional PCA. That makes plotting simpler.

# In[ ]:


from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
pca2_results = pca2.fit_transform(df.drop(["y"], axis=1))


# .... and then we plot it as scatter with the target as a color mapping.

# In[ ]:


cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots(figsize=(20,15))
points = ax.scatter(pca2_results[:,0], pca2_results[:,1], c=df.y, s=50, cmap=cmap)
f.colorbar(points)
plt.show()


# Interesting.... it looks like some pattern. Why?

# ## T-SNE  (t-distributed Stochastic Neighbor Embedding)
# This is a more modern method of dimensionality reduction. I just use it, I have no idea how it actually works. We still do reduction to two dimensions. Makes plotting simple.

# In[ ]:


from sklearn.manifold import TSNE
tsne2 = TSNE(n_components=2)
tsne2_results = tsne2.fit_transform(df.drop(["y"], axis=1))


# In[ ]:


f, ax = plt.subplots(figsize=(20,15))
points = ax.scatter(tsne2_results[:,0], tsne2_results[:,1], c=df.y, s=50, cmap=cmap)
f.colorbar(points)
plt.show()


# Ha! Even more interesting! It even looks like we can make some regression out of this set! I'll try that later.
# 
# Please upvote if you like this!
