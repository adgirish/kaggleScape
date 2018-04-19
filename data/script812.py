
# coding: utf-8

# ## This is my approach to understand the competition problem dataset. I still have to figure out using the competition API, cleaning of data, regression technique to use etc. So basically most of it ...
# 
# ### Happy to receive feedback from all of you :)

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_hdf("../input/train.h5")


# ### First look at the dataset

# In[ ]:


df.shape


# In[ ]:


df.head()


# So we have more than 100 features or variables. We can also see a bunch of missing values in our dataset. 
# 
# ***No other information has been provided through the challenge about the data.*
# 

# ### Lets remove/handle the missing data

# In[ ]:


#df.describe()


# To get a better idea of the dataset in terms of values each feature holds

# In[ ]:


df.isnull().any()


# Most of the columns look like they have null values so we can consider interpolating data. Since, no information about the dataset is available we will avoid that for and look at row wise values as of now.

# In[ ]:


#Super slow method to find all the rows with no NaN values in any of the columns

#count=0
#for x in range(len(df)-1710750):
#    if not any(df.isnull().values[x]):
#        count=count+1


# In[ ]:


df_clean=df.dropna(axis='index',how='any')
df_clean.shape


# In[ ]:


df_clean.shape[0]/df.shape[0]*100


# So only 13% of the records(timestamps) have values in all columns. Lets look how each column performs in terms of missing data. This information might help in deciding if we should interpolate missing data or directly remove it.

# In[ ]:


def count_missing(df):
    stats={}
    for x in range(len(df.columns)):        
        stats[df.columns[x]]=df[df.columns[x]].isnull().sum()/len(df[df.columns[x]])*100
    return stats


# In[ ]:


res=count_missing(df)


# In[ ]:


plt.figure(figsize=(10,25))
plt.barh(range(0,111),res.values(),align='center')
plt.yticks(range(0,111),(res.keys()))
plt.autoscale()
plt.show()


# ### What all can we do with the missing data now...
# 
# 1. Remove it   
# 2. Interpolate (but with no information about the data, it's kind of a shot in the dark)-choosing the appropriate   
# method is also tough
# 3. Anything else I should know? :/   

# Let's try to plot the ones which have low errors and see if we can identify any trends

# In[ ]:


low_err={k:v for (k,v) in res.items() if v<0.5}
del low_err['id']
del low_err['timestamp']
del low_err['y']
print((low_err),len(low_err))


# We will try to plot the features with less than 0.5% error and see if we can find any pattern in some of the features

# In[ ]:


l=list(low_err.keys())
for i in range(len(low_err)):
    plt.figure(figsize=(15,5))
    plt.scatter(y=df[l[i]],x=df['timestamp'])
    plt.title(l[i],fontsize=15)
    plt.xlabel('Timestamp Value')
    plt.ylabel('Value')
    plt.xlim(0,1900)


# ### Let's now look at the distribution of data in these features.

# In[ ]:


for i in range(len(low_err)):
    plt.figure(figsize=(15,5))
    #plt.scatter(y=df[l[i]],x=df['timestamp'])
    sns.distplot(df[l[i]].dropna().values)  #Removed NaN values for now
    plt.title(l[i],fontsize=15)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    


# ### Checking correlation of all the features (total 108 for now) with y.     
# *Note: We used df_clean and not df here.*

# In[ ]:


plt.figure(figsize=(30,20))
sns.heatmap(df_clean.corr(method='pearson', min_periods=1),cmap='RdYlGn')


# ### None of the features have a strong linear correlation with y (We checked for pearson correlation here). Spearman correation takes a lot of processing time and thus we will limit it to features which have less than 0.5% missing values in the dataset.
# 
# ### Some observation points:
# 
# 1. Features under the tag of "fundamental" have some kind of correlation among them as you can see from the box in top left quadrant of the heatmap. Similarly, "technical" features are somewhat related to each other (bottom right quadrant).    
# 2. A relatively lower correlation is seen between the two types of features. (Not a lot of greens and reds in the top right and bottom left quadrant of the heatmap).     
# 3. Some of the features have close correlation to their neighbors as can be seen from green clusters on the diagnol.
# 
# 
# 

# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(df_clean.corr(method='spearman', min_periods=1),cmap='RdYlGn')


# Spearman correlation reinforces the observation points while checking non-linear correlations as well.
# 

# Let's have a look at univariate correaltion with the prediction variable.

# In[ ]:


plt.figure(figsize=(10,25))
uni_cor=df_clean.apply(lambda x: x.corr(df["y"],method='spearman'),axis=0)
uni_cor=uni_cor.drop('y')
uni_cor.sort_values(inplace=True)
plt.barh(range(0,len(uni_cor)),uni_cor,align='center')
plt.yticks(range(0,len(uni_cor)),list(df_clean.columns.values))
plt.autoscale()
plt.show()


# As we can see the correlations are not very strong for any variable individually. So we need a to emphasize a lot on feature selection.

# ### To focus on:
# 
# 1. How to deal with missing values - Remove directly (df_clean) or use some kind of interpolation.
# 2. Which features are to be used?
# 3. Which Regeression model to use?

# ### Ok this exploration stuff is getting boring now. Lets try some regression methods and see what we get from the clean dataset (~13% of the total values). We switch to the other notebook for that.
# 
# *Will keep you guys posted when i make some progress*
