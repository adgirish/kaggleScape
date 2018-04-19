
# coding: utf-8

# # YouTube Trending Statistics Exploration in Python
# 
# This notebook will walk you through some preliminary data exploration process of the *YouTube Trending* dataset, specifically the US dataset. Original data and more information could be found on two Kaggle sites:
# - [Link 1](https://www.kaggle.com/datasnaek/youtube)
# - [Link 2](https://www.kaggle.com/datasnaek/youtube-new)

# ## Importing libraries

# In[1]:


import pandas as pd
import numpy as np

import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

matplotlib.rcParams['figure.figsize'] = (10, 10)


# ## Reading in a dataset

# In[2]:


file_name = '../input/USvideos.csv' # change this if you want to read a different dataset
my_df = pd.read_csv(file_name, index_col='video_id')
my_df.head()


# ## Processing the dates
# If we look at the `trending_date` or `publish_time` columns, we see that they are not yet in the correct format of datetime data.

# In[3]:


my_df['trending_date'] = pd.to_datetime(my_df['trending_date'], format='%y.%d.%m')
my_df['trending_date'].head()


# In[4]:


my_df['publish_time'] = pd.to_datetime(my_df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
my_df['publish_time'].head()


# In[5]:


# separates date and time into two columns from 'publish_time' column
my_df.insert(4, 'publish_date', my_df['publish_time'].dt.date)
my_df['publish_time'] = my_df['publish_time'].dt.time
my_df[['publish_date', 'publish_time']].head()


# ## Processing data types
# Some columns have their data types inappropriately registered by Pandas. For example, `views`, `likes`, and similar columns only need `int` data type, instead of `float` (to save memory), or `category_id`, a nominal attribute, should not carry `int` data type.
# 
# It is important that we ourselves assign their data types appropriately.

# In[6]:


type_int_list = ['views', 'likes', 'dislikes', 'comment_count']
for column in type_int_list:
    my_df[column] = my_df[column].astype(int)

type_str_list = ['category_id']
for column in type_str_list:
    my_df[column] = my_df[column].astype(str)


# ## Processing `category_id` column
# Here we are adding the `category` column after the `category_id` column, using the `US_category_id.json` file for lookup.

# In[7]:


# creates a dictionary that maps `category_id` to `category`
id_to_category = {}

with open('../input/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        id_to_category[category['id']] = category['snippet']['title']

id_to_category


# In[8]:


my_df.insert(4, 'category', my_df['category_id'].map(id_to_category))
my_df[['category_id', 'category']].head()


# ## Correlation analysis and heatmap

# In[9]:


keep_columns = ['views', 'likes', 'dislikes', 'comment_count'] # only looking at correlations between these variables
corr_matrix = my_df[keep_columns].corr()
corr_matrix


# In[10]:


fig, ax = plt.subplots()
heatmap = ax.imshow(corr_matrix, interpolation='nearest', cmap=cm.coolwarm)

# making the colorbar on the side
cbar_min = corr_matrix.min().min()
cbar_max = corr_matrix.max().max()
cbar = fig.colorbar(heatmap, ticks=[cbar_min, cbar_max])

# making the labels
labels = ['']
for column in keep_columns:
    labels.append(column)
    labels.append('')
ax.set_yticklabels(labels, minor=False)
ax.set_xticklabels(labels, minor=False)

plt.show()


# ## Handling videos that trended in multiple days
# 
# A number of videos appear multiple times in our dataset, as they were trending across multiple days. For our purposes, we will remove these duplicated entries for now, and only keep the last entry of each video, as that entry will have to most updated statistics of the corresponding video.

# In[11]:


print(my_df.shape)
my_df = my_df[~my_df.index.duplicated(keep='last')]
print(my_df.shape)
my_df.index.duplicated().any()


# We now have a 3719-entry dataset, and no row in that dataset is duplicated (as the `any()` function tells us). If you're interested in a time series analysis of videos that were trending in multiple days, check out my other notebook: [kaggle.com/quannguyen135/python-time-series-analysis](https://www.kaggle.com/quannguyen135/python-time-series-analysis).

# ## Visualizing _most_ statistics
# Here we want to look at the videos that have the most views, likes, dislikes, comments, etc. For convenience purposes, we will write a function that would take in a column name and visualize the videos that have the most counts for statistics specified by the column name.

# In[12]:


def visualize_most(my_df, column, num=10): # getting the top 10 videos by default
    sorted_df = my_df.sort_values(column, ascending=False).iloc[:num]
    
    ax = sorted_df[column].plot.bar()
    
    # customizes the video titles, for asthetic purposes for the bar chart
    labels = []
    for item in sorted_df['title']:
        labels.append(item[:10] + '...')
    ax.set_xticklabels(labels, rotation=45, fontsize=10)
    
    plt.show()


# Now we could call the `visualize_most()` function while passing different column names:

# In[13]:


visualize_most(my_df, 'views')


# In[14]:


visualize_most(my_df, 'likes', num=5) # only visualizes the top 5


# In[15]:


visualize_most(my_df, 'dislikes')


# In[16]:


visualize_most(my_df, 'comment_count')


# ## Video-specific statistics visualizations
# Sometimes we might want to look to statistics on specific videos, and compare them against each other. Here we will also be writing functions so that we could dynamically call them on different parameters later.
# 
# The first function will visualize the statistics next to each other, and the second will visualize `likes` and `dislikes` stacked.

# In[17]:


def visualize_statistics(my_df, id_list): # taking a list of video ids
    target_df = my_df.loc[id_list]
    
    ax = target_df[['views', 'likes', 'dislikes', 'comment_count']].plot.bar()
    
    # customizes the video titles, for asthetic purposes for the bar chart
    labels = []
    for item in target_df['title']:
        labels.append(item[:10] + '...')
    ax.set_xticklabels(labels, rotation=45, fontsize=10)
    
    plt.show()

def visualize_like_dislike(my_df, id_list):
    target_df = my_df.loc[id_list]
    
    ax = target_df[['likes', 'dislikes']].plot.bar(stacked=True)
    
    # customizes the video titles, for asthetic purposes for the bar chart
    labels = []
    for item in target_df['title']:
        labels.append(item[:10] + '...')
    ax.set_xticklabels(labels, rotation=45, fontsize=10)
    
    plt.show()


# Next, we will generate a random sample from our dataset, but you could always pass in a list of the IDs of the videos you specifically want to look at.

# In[18]:


sample_id_list = my_df.sample(n=10, random_state=4).index # creates a random sample of 10 video IDs
sample_id_list


# In[19]:


visualize_statistics(my_df, sample_id_list)


# In[20]:


visualize_like_dislike(my_df, sample_id_list)


# ## Histograms

# - Individual histograms:

# In[21]:


my_df['dislikes'].plot.hist()

plt.show()


# In[22]:


my_df['comment_count'].plot.hist()

plt.show()


# - Different histograms together:

# In[23]:


my_df[['likes', 'dislikes']].plot.hist(alpha=0.5)

plt.show()


# In[24]:


my_df[['dislikes', 'comment_count']].plot.hist(alpha=0.5)

plt.show()


# ## Category analysis
# Here we want to look at how popular each category is among top trending videos

# In[25]:


category_count = my_df['category'].value_counts() # frequency for each category
category_count


# In[26]:


ax = category_count.plot.bar()
ax.set_xticklabels(labels=category_count.index, rotation=45, fontsize=10)

plt.show()

