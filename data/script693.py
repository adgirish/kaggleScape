
# coding: utf-8

# ## Global Imports ##

# In[ ]:




import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading Training Dataset ##

# In[ ]:


data = pd.read_json("../input/train.json")


# ## Lets Describe Data A Bit ##

# In[ ]:


data.head()


# In[ ]:


data.describe()


# ## Lets See If We Have Any Missing Values In The Data ##

# In[ ]:


msno.matrix(data,figsize=(13,3))


# The data looks clean as there are no missing values in the any column

# ## Visualizing Distribution Of Price Before and After Removing Outliers ##

# In[ ]:


dataPriceLimited = data.copy()
upperLimit = np.percentile(dataPriceLimited.price.values, 99)
dataPriceLimited['price'].ix[dataPriceLimited['price']>upperLimit] = upperLimit
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(13,5)
sn.distplot(data.price.values, bins=50, kde=True,ax=ax1)
sn.distplot(dataPriceLimited.price.values, bins=50, kde=True,ax=ax2)


# ## Visualizing Outliers In Data ##
# Lets understand what category of interest level contribute more to outliers in price

# In[ ]:


fig, (axes) = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(13, 8)
sn.boxplot(data=data,y="price",orient="v",ax=axes[0][0])
sn.boxplot(data=data,y="price",x="interest_level",orient="v",ax=axes[0][1])
sn.boxplot(data=dataPriceLimited,y="price",orient="v",ax=axes[1][0])
sn.boxplot(data=dataPriceLimited,y="price",x="interest_level",orient="v",ax=axes[1][1])


# The price contains few outliers which skews the distribution towards the right. But when we split the data by interest level it is clearly visible the skewness is purely caused by price points in 'Low' interest level.

# ## Visualizing Interest Level Vs Price ##

# In[ ]:


fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(13,5)

interestGroupedData = pd.DataFrame(data.groupby("interest_level")["price"].mean()).reset_index()
interestGroupedSortedData = interestGroupedData.sort_values(by="price",ascending=False)
sn.barplot(data=interestGroupedSortedData,x="interest_level",y="price",ax=ax1,orient="v")
ax1.set(xlabel='Interest Level', ylabel='Average Price',title="Average Price Across Interest Level")

interestData = pd.DataFrame(data.interest_level.value_counts())
interestData["interest_level_original"] = interestData.index
sn.barplot(data=interestData,x="interest_level_original",y="interest_level",ax=ax2,orient="v")
ax2.set(xlabel='Interest Level', ylabel='Interest Level Frequency',title= "Frequency By Interest Level")


# It can be clearly visible from above two graphs
# 
#  1. People showed low interest to high priced rental Listing and vice versa
#  2. Distribution of dataset by interest level contains more of interest level "Low"

# ## Visuallizing Interest Level Vs Bathroom ##

# In[ ]:


fig,(ax1,ax2)= plt.subplots(nrows=2)
fig.set_size_inches(13,8)

sn.countplot(x="bathrooms", data=data,ax=ax1)
data1 = data.groupby(['bathrooms', 'interest_level'])['bathrooms'].count().unstack('interest_level').fillna(0)
data1[['low','medium',"high"]].plot(kind='bar', stacked=True,ax=ax2)


# ## Visualizing Interest Level Vs Bedrooms ##

# In[ ]:


fig,(ax1,ax2)= plt.subplots(nrows=2)
fig.set_size_inches(13,8)

sn.countplot(x="bedrooms", data=data,ax=ax1)
data1 = data.groupby(['bedrooms', 'interest_level'])['bedrooms'].count().unstack('interest_level').fillna(0)
data1[['low','medium',"high"]].plot(kind='bar', stacked=True,ax=ax2)


# ## Visualizing Interest Level Vs Hour ##

# In[ ]:


data["created"] = pd.to_datetime(data["created"])
data["hour"] = data["created"].dt.hour
fig,(ax1,ax2)= plt.subplots(nrows=2)
fig.set_size_inches(13,8)

sn.countplot(x="hour", data=data,ax=ax1)

data1 = data.groupby(['hour', 'interest_level'])['hour'].count().unstack('interest_level').fillna(0)
data1[['low','medium',"high"]].plot(kind='bar', stacked=True,ax=ax2,)


# ## Bedrooms Vs Bathrooms Vs Interest Level ##

# In[ ]:


fig,(ax1)= plt.subplots()
fig.set_size_inches(13,8)
ax1.scatter(data[data['interest_level']=="low"]['bedrooms'],data[data['interest_level']=="low"]['bathrooms'],c='green',s=40)
ax1.scatter(data[data['interest_level']=="medium"]['bedrooms'],data[data['interest_level']=="medium"]['bathrooms'],c='red',s=40)
ax1.scatter(data[data['interest_level']=="high"]['bedrooms'],data[data['interest_level']=="high"]['bathrooms'],c='blue',s=80)
ax1.set_xlabel('Bedrooms')
ax1.set_ylabel('Bathrooms')
ax1.legend(('Low','Medium','High'),scatterpoints=1,loc='upper right',fontsize=15,)


# It can be visible from the above chart people show "High" interest level when the  no of bedrooms are on par with no of bathrooms. The blue diagonal dots gives evidence for it.

# ## Correlation Between Price and Other Features ##
# 
# Since price has got high impact with Interest Level. It is interesting to understand what other features correlate with price

# In[ ]:


corrMatt = data[["bedrooms","bathrooms","price"]].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)


# Bedroom and Bathroom has got less correlation with price. But it is common than price of the property tend to increase as the no of bathroom and bedroom increases. It looks some other variable like location (latitude and longitude) and feature may got high impact on price than the above features.

# ## Kindly Upvote if you like the notebook  ##
