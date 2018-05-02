
# coding: utf-8

# Simple notebook to explore the variables present in the dataset. 
# 
# Please upvote if you find it useful :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 120)


# Please note that, in this competition HDF5 file is being used instead of csv.

# In[ ]:


with pd.HDFStore("../input/train.h5", "r") as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")


# **Sneak-peek at the data:**
# 
# Let us look at the top few rows to understand the variables and the nature of data

# In[ ]:


df.head()


# So there are 111 columns present in the dataset.
# 
# - 1 id column
# - 1 timestamp column
# - 5 columns with name prefix 'derived'
# - 63 columns with name prefix 'fundamental' - fundamental_0 to fundamental_63 - 'fundamental_4' is missing. Any specific reasons?
# - 40 columns with name prefix 'technical' - technical_0 to technical_44 - technical_4, technical_8, technical_15, technical_23, technical_26 are missing.  
# - 1 target variable named 'y'
# 
# Now let us look at the distribution of data in each of these columns

# In[ ]:


df.describe()


# It seems NaN values are present in all input columns but  for two (technical_22 and technical_34).
# 
# So let us count the number of missing values in each of the columns.

# In[ ]:


labels = []
values = []
for col in df.columns:
    labels.append(col)
    values.append(df[col].isnull().sum())
    print(col, values[-1])


# In[ ]:


ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,50))
rects = ax.barh(ind, np.array(values), color='y')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
#autolabel(rects)
plt.show()


# Fundamental_5 has the most number of missing values followed by fundamental_38.
# 
# **Distribution plot:**
# 
# Now let us look at the distribution plot of some of the numeric variables. 
# 
# Univariate analysis from [this notebook][1] reveals some important variables. So let us look at the plots of  top 4 variables.
# 
# - technical_30
# - technical_20
# - fundamental_11
# - technical_19
# 
#   [1]: https://www.kaggle.com/sudalairajkumar/two-sigma-financial-modeling/univariate-analysis-regression-lb-0-006

# In[ ]:


cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']
fig = plt.figure(figsize=(8, 20))
plot_count = 0
for col in cols_to_use:
    plot_count += 1
    plt.subplot(4, 1, plot_count)
    plt.scatter(range(df.shape[0]), df[col].values)
    plt.title("Distribution of "+col)
plt.show()


# Some of the observations from the distribution plot are:
# 
# - The top two variables (technical_30 and technical_20) range between 0 and 0.8 and there are no major outliers
# - Fundamental_11 has few outliers at the beginning and then looks more or less fine with two small peaks
# - Technical_19 has few high values towards the end
# 
# **Target Distribution:**
# 
# Now let us scatter plot the target variable.
# 
# 

# In[ ]:


plt.figure(figsize=(8, 5))
plt.scatter(range(df.shape[0]), df.y.values)
plt.show()


# Target values range between -0.086 to 0.093. 
# 
# As we can see the target graph is more darker at the middle, suggesting more values are concentrated in those region. 
# 
# Also there seems to be some hard stop at both the ends (probably capping the target to remain within the limits?!) - this could be inferred from the two dark lines at the top and bottom.
# 
# Also there seems to be some change in the target distribution with respect to time. As we move from left to right, initially the target is evenly distributed in the given range (-0.08 to 0.09) and then in the middle it is not so.
# 
# **Timestamp:**
# 
# Now let us look at the counts for each of the timestamps present in the data.

# In[ ]:


fig = plt.figure(figsize=(12, 6))
sns.countplot(x='timestamp', data=df)
plt.show()


# So there is an increasing trend in the number of rows for each of the time stamps. Also there are some sudden jumps in between at intervals. 
# 
# To know more, please refer to this excellent kernel by @anokas.
# 
# From this [kernel][1], it seems ids are the assets that we are tracking.
# 
# So let us see the number of assets that are present in the data.
# 
# 
#   [1]: https://www.kaggle.com/ysidhu/two-sigma-financial-modeling/two-sigma-portfolio-returns-eda

# In[ ]:


print(len(df.id.unique()))


# So we have 1424 unique assets in the dataset. As we can see from the previous plot of timestamp, ~1100 assets is the maximum number of assets at any given timestamp. So there are few assets that are dropped in between.
# 
# Now we can check the 'y' distribution of some of the assets. Let us first look at ids with high negative mean target values. 

# In[ ]:


temp_df = df.groupby('id')['y'].agg('mean').reset_index().sort_values(by='y')
temp_df.head()


# In[ ]:


id_to_use = [1431, 93, 882, 1637, 1118]
fig = plt.figure(figsize=(8, 25))
plot_count = 0
for id_val in id_to_use:
    plot_count += 1
    plt.subplot(5, 1, plot_count)
    temp_df = df.ix[df['id']==id_val,:]
    plt.plot(temp_df.timestamp.values, temp_df.y.values)
    plt.plot(temp_df.timestamp.values, temp_df.y.cumsum())
    plt.title("Asset ID : "+str(id_val))
    
plt.show()


# Blue line represents the distribution of 'y' variable in the given time stamp. Green line represents the cumulative 'y' value
# 
# So 4 out these 5 assets are dropped (as they are not present till the last time stamp which is 1812), when the cumulative negative target value falls steeply. 
# 
# Now let us take the assets with high positive mean target value and see their distribution.

# In[ ]:


temp_df = df.groupby('id')['y'].agg('mean').reset_index().sort_values(by='y')
temp_df.tail()


# In[ ]:


id_to_use = [767, 226, 824, 1809, 1089]
fig = plt.figure(figsize=(8, 25))
plot_count = 0
for id_val in id_to_use:
    plot_count += 1
    plt.subplot(5, 1, plot_count)
    temp_df = df.ix[df['id']==id_val,:]
    plt.plot(temp_df.timestamp.values, temp_df.y.values)
    plt.plot(temp_df.timestamp.values, temp_df.y.cumsum())
    plt.title("Asset ID : "+str(id_val))
plt.show()


# Interestingly 2 of these 5 good performing assets are also dropped (Assets 824 and 1089). Not sure about the reasons though.
# 
# Now let us take some assets which are present across all the timestamps and see their distribution.

# In[ ]:


temp_df = df.groupby('id')['y'].agg('count').reset_index().sort_values(by='y')
temp_df.tail()


# In[ ]:


id_to_use = [1548, 699, 697, 704, 1066]
fig = plt.figure(figsize=(8, 25))
plot_count = 0
for id_val in id_to_use:
    plot_count += 1
    plt.subplot(5, 1, plot_count)
    temp_df = df.ix[df['id']==id_val,:]
    plt.plot(temp_df.timestamp.values, temp_df.y.values)
    plt.plot(temp_df.timestamp.values, temp_df.y.cumsum())
    plt.title("Asset ID : "+str(id_val))
plt.show()


# Asset 699 looks like a very good asset.! 
# 
# To know more about assets, refer to this [excellent kernel][1] by omolluska.
# 
# Happy Kaggling.!
# 
# 
#   [1]: https://www.kaggle.com/sankhamukherjee/two-sigma-financial-modeling/when-why-are-stocks-bought-and-sold
