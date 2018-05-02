
# coding: utf-8

# Given that Id's correspond to different assets, it's possible that features have different relations with different Id's. Some features may be important for one group of Id's and may be not important for another group. Let's try this idea simply by calculating correlations of all features with target for each Id separately. 
# 
# Heatmap in the end of the notebook is the main visualization.

# In[ ]:


# get the data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")
    
df.head()


# Firstly let's look at correlations with target for all data. They are very small.

# In[ ]:


corr = df.iloc[:,2:-1].corrwith(df.y)
print('max_correlation', corr.max().max())
print('min_correlation', corr.min().min())


# In[ ]:


plt.figure(figsize=(5,15))
corr.plot(kind='barh')


# What number of timestamps for each Id do we have in dataset?

# In[ ]:


df.groupby('id').size().hist(bins=50)


# In[ ]:


df.groupby('id').size().value_counts().head(5)


# There are Id's with very different number of timestamps. It's not correct to compare for example correlations for Id with 50 points and Id with 1800 points because Id with 50 points could have higher correlation just by chance due to variability in small samples.
# 
# For simplicity let's take only large group with maximum number of observations.

# In[ ]:


size = df.groupby('id').size()
print(len(size), max(size))
assets = size[size==1813].index.values
print(len(assets))

correlations = pd.DataFrame()

for asset in assets:
    
    df2 = df[df.id==asset]
    corr = df2.drop(['id', 'timestamp', 'y'], axis=1).corrwith(df2.y)
    correlations[asset] = corr

correlations.head()


# In[ ]:


print('max_correlation', correlations.max().max())
print('min_correlation', correlations.min().min())


# Distribution of maximum correlation per Id

# In[ ]:


correlations.max().hist(bins=50)


# Minunum and maximum correlation per feature

# In[ ]:


plt.figure(figsize=(8,15))
ax1 = plt.subplot(121)
correlations.min(axis=1).plot(kind='barh')
plt.subplot(122, sharey=ax1)
correlations.max(axis=1).plot(kind='barh')


# Correlation heatmap - strength of correlations of all features with all Id's - Id's are on x-axis and features are on y-axis.

# In[ ]:


plt.figure(figsize=(8,15))
sns.heatmap(correlations, vmin=-0.22, vmax=0.22)


# As we see, there are some features for which correlation is quite stable across Id's (for example, technical_20, techical_30, fundametal_15, fundamental_57). There are some features for which correlation is high for only small number of Id's (fundamental_1, fundamental_28, fundamental_61).
# 
# For most of features there are Id's with both positive and negative correlations. Even for most stable features there are several Id's with opposite sign. So this is the reason why overall correlations for all data are very small.
# 
# 
# P.S. The question of missing values wasn't taken into consideration. Maybe for some Id-feature pairs there are a lot of missing values and hence small number of observations.
