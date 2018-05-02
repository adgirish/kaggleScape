
# coding: utf-8

# This is an initial exploration of the dataset to find the parameters which has a strong correlation with the target variable. 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.plotly as py
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 120)


# In[ ]:


# read-in the training data
with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")


# In[ ]:


# I will keep a copy of the dataframe for later use
df1 = df.copy()


# In[ ]:


df.head()


# In[ ]:


# let's check the size of the dataframe
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# In[ ]:


# Extract the basic statistics 
df.describe()


# **Number of the missing values**

# In[ ]:


# Let's look how many missing values in each column 
# We will sort missing values to bring column with the highest number of missing values at the top
# We will see the number of missing values in each of the first 30 columns

df.isnull().sum().sort(axis=0, ascending=False, inplace=False).head(30)


# Since we have a lot of missing values, we should find a strategy to replace them with some correlated values

# In[ ]:


# But for now I am going to inpute any missing values with the mean
# I will deal with this major issue later on

df = df.fillna(df.mean()['derived_0':'y'])


# In[ ]:


# Double check the missing values
df.isnull().sum().head(30)


# In[ ]:


# Let's see the distribution of the target variable

df["y"].hist(bins = 30, color = "orange")
plt.xlabel("Target Variable")
plt.ylabel("Frequency")


# In[ ]:


# Take absolute values 
#df.loc[:, "derived_0": "technical_44"] = df.loc[:, "derived_0": "technical_44"].abs()


# In[ ]:


# Groupby the target variable
df_f = (df.groupby(pd.cut(df["y"], [-0.087,-0.067,-0.047,-0.027,-0.007,0.013,0.033,0.053,0.073,0.094], right=False))
        .mean())


# In[ ]:


df_f.head()


# In[ ]:


# The correlation matrix
cor_mat = df_f.corr(method='pearson', min_periods=1).sort(axis=0, ascending=False, inplace=False)
cor_mat.head(20) # Look at the first 20 columns


# In[ ]:


# The correlation with the target variable sorted in a descending order
# Look at the first 30 parameters
cor_mat.iloc[0,:].sort(axis=0, ascending=False, inplace=False).head(30) 


# Let's plot the most correlated variables with the target variable
# The most correlated variables in a descending order:
# technical_0"  
# technical_24  
# technical_44

# In[ ]:


alpha = plt.figure()
plt.scatter(df_f["technical_0"], df_f["y"], alpha=.1, s=400)
plt.xlabel("technical_0") 
plt.ylabel("Target variable")
plt.show()


# In[ ]:


alpha = plt.figure()
plt.scatter(df_f["technical_24"], df_f["y"], alpha=.1, s=400)
plt.xlabel("technical_24") 
plt.ylabel("Target variable")
plt.show()


# In[ ]:


alpha = plt.figure()
plt.scatter(df_f["technical_44"], df_f["y"], alpha=.1, s=400)
plt.xlabel("technical_44") 
plt.ylabel("Target variable")
plt.show()


# The correlation is not clear yet and this is maybe due to the issue of the missing values.
# I am going to remove the missing values once to check the correlations with the target variables in case we used only clean data without missing values.

# **Remove the missing values**

# In[ ]:


df_a = df1.dropna(axis=0)


# In[ ]:


len(df_a)


# We have 223040 observations left after we removed the missing values. Let's use this clean data for exploration and later I will deal with the issue of the missing values in a reasonable way.

# In[ ]:


# Again groupby the target variable
df_f = (df_a.groupby(pd.cut(df_a["y"], [-0.087,-0.067,-0.047,-0.027,-0.007,0.013,0.033,0.053,0.073,0.094], right=False))
        .mean())


# In[ ]:


# The correlation matrix
cor_mat = df_f.corr(method='pearson', min_periods=1).sort(axis=0, ascending=False, inplace=False)
cor_mat.head(20)


# In[ ]:


# The correlation with the target variable sorted in a descending order
cor_mat.iloc[0,:].sort(axis=0, ascending=False, inplace=False).head(20)


# Nice! We have got a better correlation than before. It seems that inputting the missing values with the means has degraded the correlation with the target variable.

# Let's plot the most correlated variables with the target variable. Still the most correlated variables are the same even after the missing data is removed. The most correlated variables in a descending order:
# technical_0"
# technical_24
# technical_44

# In[ ]:


cor_matt = cor_mat.iloc[0,:].sort(axis=0, ascending=False, inplace=False)


# In[ ]:


# Extract the most 10 correlated variables
cor_matt = cor_matt.keys()[1:10]


# In[ ]:


for i in cor_matt:
    alpha = plt.figure()
    plt.scatter(df_f[i], df_f["y"], alpha=.1, s=400)
    plt.xlabel(i) 
    plt.ylabel("Target Variable")
    plt.show()


# In[ ]:


# Let's take a 1000 sample of the data to explore 
# We will use raw data which has the missing data removed from it
df_m = df_a.sample(n=1000)


# In[ ]:


# Plot the most correlated variables 
for i in cor_matt:
    alpha = plt.figure()
    plt.scatter(df_m["timestamp"], df_m[i], alpha=.5)
    plt.xlabel("timestamp") 
    plt.ylabel(i)
    plt.show()


# **Thanks for visiting. This was a first glimpse on the data and more is coming shortly**
