
# coding: utf-8

# # INTRODUCTION
# 
# The main aim of this notebook is going to be a very high-level Exploratory Data Analysis and Visualisation of the data, just to get a feel for the numbers and to play around with some of Seaborn's as well as wordcloud's plotting capabilities. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')


# let's start by importing the crime dataset and looking at the first 5 rows to get a feel

# In[ ]:


crime = pd.read_csv('../input/Crime1.csv')
crime.head()


# It's quite a small and concise dataset but there should be quite a lot that can be done with regards to EDA on this. So let's get started. I will start from the left to the right with the "Category" column

# #Which are the most common categories of crime committed?
# 
# Here, we will take a quick look at some summary aggregates to see what was the most common category of crime that was registered or committed. I will use pandas convenient "value_counts()" method which outputs the volume (number of) crimes grouped per category

# In[ ]:


crime.Category.value_counts()


# In[ ]:


# Create a dataframe containing the Category counts
category = pd.DataFrame(list(zip(crime.Category.value_counts().index,crime.Category.value_counts())), columns=['Category','value'], index=None)


# Let's create a factorplot as well as a wordcloud of the different categories for some nice visualisation

# In[ ]:


# Generating the wordcloud with the values under the category dataframe
catcloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=1200,
                          height=800
                         ).generate(" ".join(category['Category'].values))


# In[ ]:


plt.imshow(catcloud, alpha=0.8)
plt.axis('off')
plt.show()


# In[ ]:


# Generating the factorplot
sns.factorplot(x='value', y = 'Category', data=category,kind="bar", size=4.25, aspect=1.9, palette="cubehelix")
plt.title('Factorplot of the category of crime and number of occurences ')


# As we can see from the above factorplots, "larceny/theft", "non-criminal" and "other offenses" were the chief crime category culprits

# # Most common crimes carried out per it's description?
# 
# The code and visualisation here in this section will be carried out much like in the previous one where we will also look at summary aggregates of the counts per description type

# In[ ]:


crime.Descript.value_counts()


# This column contains a lot more detailed information about the type of the crime committed and right-away we can observe that Grand Theft Auto is the most common in this area. well well..
# 
# As again, we create another dataframe which will make it convenient for the plotting 

# In[ ]:


descript = pd.DataFrame(list(zip(crime.Descript.value_counts().index,crime.Descript.value_counts())), columns=['Description','counts'], index=None)


# In[ ]:


descloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=1500,
                          height=1400
                         ).generate(" ".join(descript['Description'].values))


# In[ ]:


plt.imshow(descloud,alpha=0.8)
plt.axis('off')
plt.show()


# #Day on which there is most crime?
# 
# How about looking on which day criminals most want to come out to play?

# In[ ]:


DOW = pd.DataFrame(list(zip(crime.DayOfWeek.value_counts(),crime.DayOfWeek.value_counts().index)), columns=['count','Day'], index=None)


# In[ ]:


sns.factorplot(x="count", y="Day", data = DOW, kind="bar", size=4, aspect=1.9)


# Interestingly we see that it is a Saturday when most crime is committed followed by a Thursday. 

# # Which are the most crime-ridden Districts?

# In[ ]:


crime.PdDistrict.value_counts()


# In[ ]:


district = pd.DataFrame(list(zip(crime.PdDistrict.value_counts().index,crime.PdDistrict.value_counts())), columns=['District','count'], index=None)


# In[ ]:


sns.factorplot(x="count", y="District", data = district, kind="bar", size=4, aspect=1.9, palette='PuBuGn_d')


# Well looks like the top 3 crime-ridden areas are Southern, Northern and Mission districts. 

# Let's go onto the "Resolution" column and look at the summary statistics. Hopefully we find that many of these crimes have been successfully "resolved"

# In[ ]:


# Create the dataframe just for the Resolution data and the aggregation
Resolution = pd.DataFrame(list(zip(crime.Resolution.value_counts().index,crime.Resolution.value_counts())), columns=['resolution','value'], index=None)


# In[ ]:


rescloud = WordCloud(
                          stopwords=STOPWORDS,
                          background_color='black',
                          width=1500,
                          height=1400
                         ).generate(" ".join(Resolution['resolution'].values))


# In[ ]:


plt.imshow(rescloud, alpha=0.8)
plt.axis('off')
plt.show()


# Let's look at the factorplot of the resolution data

# In[ ]:


sns.factorplot(x='value' , y = 'resolution', data=Resolution, kind="bar", size=3.25, aspect=2.5, palette='BuGn_r')


# Oh man. This doesn't bode well as most of the crimes were not resolved.  This means that there are still quite a lot of outstanding crime cases pending.

# Anyway, we have now reached the last two columns of the dataset, X and Y. These columns are coordinates, coordinates which relate to the "address" column. Just for the purposes of this notebook, I will be plotting these coordinates via lag-plots just to visually investigate if the data is random or not. 
# 
# Refer to the Pandas visualisation webpage for a more detailed explanation: http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html

# In[ ]:


# Importing the lag_plot plotting function
from pandas.tools.plotting import lag_plot
# Lag_plot for X coordinate
plt.figure()
lag_plot(crime.X)


# In[ ]:


lag_plot(crime.Y, c='goldenrod')


# And finally let us look at the autocorrelation plot to look at the X and Y data just to check for randomness in the data over time. If the data is non-random, then one or more of the autocorrelations will be significantly non-zero, taking into account the confidence bands ( dashed and solid lines)

# In[ ]:


from pandas.tools.plotting import autocorrelation_plot
autocorrelation_plot(crime.X, color='k', marker='.', linewidth='0.25')
autocorrelation_plot(crime.Y, color='goldenrod',marker='.', linewidth='0.15')
plt.ylim(-0.15,0.15)


# Seems pretty random for this time-series data. Anyway this notebook is a work in progress.

# TO BE CONTINUED ...
