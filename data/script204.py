
# coding: utf-8

# ### *Use this one simple trick to get to the top of the leaderboard - Grandmasters hate him!*
# 
# Here's quite a high-level EDA - since the data is so huge, we want to get a better understanding of what we actually have.
# 
# If this EDA helps you, make sure to leave an upvote to motivate me to make more! :)
# 
# First off: What files do we have?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc # We're gonna be clearing memory a lot
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

p = sns.color_palette()

print('# File sizes')
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')


# Wow, that's a lot of data! Let's start off by looking at the clicks_train.csv and clicks_test.csv files, as these contain the main things we need.
# 
# Each display has a certain number of adverts. Let's look at the distribution of these advert counts, and see if they are consistent between train and test.

# In[ ]:


df_train = pd.read_csv('../input/clicks_train.csv')
df_test = pd.read_csv('../input/clicks_test.csv')


# In[ ]:


sizes_train = df_train.groupby('display_id')['ad_id'].count().value_counts()
sizes_test = df_test.groupby('display_id')['ad_id'].count().value_counts()
sizes_train = sizes_train / np.sum(sizes_train)
sizes_test = sizes_test / np.sum(sizes_test)

plt.figure(figsize=(12,4))
sns.barplot(sizes_train.index, sizes_train.values, alpha=0.8, color=p[0], label='train')
sns.barplot(sizes_test.index, sizes_test.values, alpha=0.6, color=p[1], label='test')
plt.legend()
plt.xlabel('Number of Ads in display', fontsize=12)
plt.ylabel('Proportion of set', fontsize=12)


# This looks like a perfect split to me! So we can assume the distribution to be the same within the sets - no weird trickery going on here.
# 
# What about adverts? How many adverts are there that are very often used, and how many are rare?

# In[ ]:


ad_usage_train = df_train.groupby('ad_id')['ad_id'].count()

for i in [2, 10, 50, 100, 1000]:
    print('Ads that appear less than {} times: {}%'.format(i, round((ad_usage_train < i).mean() * 100, 2)))

plt.figure(figsize=(12, 6))
plt.hist(ad_usage_train.values, bins=50, log=True)
plt.xlabel('Number of times ad appeared', fontsize=12)
plt.ylabel('log(Count of displays with ad)', fontsize=12)
plt.show()


# Here we can see that a **huge** number of ads appear just a few times in the training set (so much that we have to use a log graph to show it), with two-thirds having less than 10 appearances. This shows us that we have to be able to predict whether someone will click on an ad not just based on the past data of that specific ad, but also by linking it to other adverts.
# 
# I think this is the underlying challenge that Outbrain has for us.
# 
# Before we move on, let's check how many ads in the test set are not in the training set.

# In[ ]:


ad_prop = len(set(df_test.ad_id.unique()).intersection(df_train.ad_id.unique())) / len(df_test.ad_id.unique())
print('Proportion of test ads in test that are in training: {}%'.format(round(ad_prop * 100, 2)))


# This number is a little more reasonable, with 88% of ads appearing in both sets.
# 
# ## Events
# 
# Let's move on to the events file. I'm not going to cover the [timestamp](https://www.kaggle.com/joconnor/outbrain-click-prediction/date-exploration-and-train-test-split) or the [location](https://www.kaggle.com/andreyg/outbrain-click-prediction/explore-user-base-by-geo), since these have already been beautifully explored - you can click the links to view those EDAs.

# In[ ]:


try:del df_train,df_test # Being nice to Azure
except:pass;gc.collect()

events = pd.read_csv('../input/events.csv')
print('Shape:', events.shape)
print('Columns', events.columns.tolist())
events.head()


# In[ ]:


plat = events.platform.value_counts()

print(plat)
print('\nUnique values of platform:', events.platform.unique())


# This is very interesting, notice how 1, 2 and 3 are repeated twice in the platform, once as floats, and once as strings.
# 
# This might become useful in the future (leak? :P) - but for now, I'm just going to treat them as the same thing.

# In[ ]:


events.platform = events.platform.astype(str)
plat = events.platform.value_counts()

plt.figure(figsize=(12,4))
sns.barplot(plat.index, plat.values, alpha=0.8, color=p[2])
plt.xlabel('Platform', fontsize=12)
plt.ylabel('Occurence count', fontsize=12)


# It's still unclear what the platform means, but it's possible that it's things like computers, phones, tablets etc. 
# 
# The `\\N`s and string numbers may have come from the way that a file was parsed while creating the dataset, but it's still a mystery. If anyone has any other ideas, I'd love to hear them!
# 
# Let's do some quick analysis on the UUID next.

# In[ ]:


uuid_counts = events.groupby('uuid')['uuid'].count().sort_values()

print(uuid_counts.tail())

for i in [2, 5, 10]:
    print('Users that appear less than {} times: {}%'.format(i, round((uuid_counts < i).mean() * 100, 2)))
    
plt.figure(figsize=(12, 4))
plt.hist(uuid_counts.values, bins=50, log=True)
plt.xlabel('Number of times user appeared in set', fontsize=12)
plt.ylabel('log(Count of users)', fontsize=12)
plt.show()


# Here we see a distribution much like the ad ids, with 88% of users being unique - there will be little scope of building user-based recommendation profiles here.
# 
# I'd love to look at things like whether the same user ever clicks on the same ad twice, or whether a user transverses the training & testing set, but we're limited by the Kernels memory limit here :(
# 
# ## Categories
# 
# Outbrain has some content classification algorithms, and they have provided us with the output from these classifications. Let's take a look at some of the most popular classifications.

# In[ ]:


try:del events
except:pass;gc.collect()

topics = pd.read_csv('../input/documents_topics.csv')
print('Columns:',topics.columns.tolist())
print('Number of unique topics:', len(topics.topic_id.unique()))

topics.head()


# In[ ]:


topic_ids = topics.groupby('topic_id')['confidence_level'].count().sort_values()

for i in [10000, 50000, 100000, 200000]:
    print('Number of topics that appear more than {} times: {}'
          .format(i, (topic_ids > i).sum()))

plt.figure(figsize=(12, 4))
sns.barplot(topic_ids.index, topic_ids.values, order=topic_ids.index, alpha=1, color=p[5])
plt.xlabel('Document Topics', fontsize=12)
plt.ylabel('Total occurences', fontsize=12)
plt.show()


# In[ ]:


cat = pd.read_csv('../input/documents_categories.csv')
print('Columns:', cat.columns.tolist())
print('Number of unique categories:', len(cat.category_id.unique()))

cat_ids = cat.groupby('category_id')['confidence_level'].count().sort_values()

for i in [1000, 10000, 50000, 100000]:
    print('Number of categories that appear more than {} times: {}'
          .format(i, (cat_ids > i).sum()))

plt.figure(figsize=(12, 4))
sns.barplot(cat_ids.index, cat_ids.values, order=cat_ids.index, alpha=1, color=p[3])
plt.xlabel('Document Categories', fontsize=12)
plt.ylabel('Total occurences', fontsize=12)
plt.show()


# That's it for today, folks!
# 
# Hopefully this helps you think of some ideas - I'll continue updating this as I do more exploration over the next few days.
# 
# Once again, please upvote if this was useful :P
