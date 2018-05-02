
# coding: utf-8

# # TalkingData AdTracking Fraud Detection Challenge
# 
# TalkingData is back with another competition: This time, our task is to predict where a click on some advertising is fraudlent given a few basic attributes about the device that made the click. What sets this competition apart is the sheer scale of the dataset: with 240 million rows it might be the biggest one I've seen on Kaggle so far.
# 
# There are some similarities with the last competition TalkingData launched: https://www.kaggle.com/c/talkingdata-mobile-user-demographics - that competition was about predicting the demographics of a user given their activity, and you can view this as a similar problem (predicting whether a user is real or not given their activity). However, that competition was plagued by a [leak](https://www.kaggle.com/wiki/Leakage) where the dataset wasn't sorted properly and certain portions of the dataset had different demographic distribtions. This meant that by adding the row ID as a feature you could get a huge boost in performance. Let's hope TalkingData have learnt their lesson this time around. 😉
# 
# Looking at the evaluation page, we can see that the evaluation metric used is** ROC-AUC** (the area under a curve on a Receiver Operator Characteristic graph).
# In english, this means a few important things:
# * This competition is a **binary classification** problem - i.e. our target variable is a binary attribute (Is the user making the click fraudlent or not?) and our goal is to classify users into "fraudlent" or "not fraudlent" as well as possible
# * Unlike metrics such as [LogLoss](http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/), the AUC score only depends on **how well you well you can separate the two classes**. In practice, this means that only the order of your predictions matter,
#     * As a result of this, any rescaling done to your model's output probabilities will have no effect on your score. In some other competitions, adding a constant or multiplier to your predictions to rescale it to the distribution can help but that doesn't apply here.
#   
# If you want a more intuitive explanation of how AUC works, I recommend [this post](https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it).
# 
# Let's dive right in by looking at the data we're given:

# In[18]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mlcrate as mlc
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pal = sns.color_palette()

print('# File sizes')
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')


# Wow, that is some really big data. Unfortunately we don't have enough kernel memory to load the full dataset into memory; however we can get a glimpse at some of the statistics:

# In[19]:


import subprocess
print('# Line count:')
for file in ['train.csv', 'test.csv', 'train_sample.csv']:
    lines = subprocess.run(['wc', '-l', '../input/{}'.format(file)], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(lines, end='', flush=True)


# That makes **185 million rows** in the training set and ** 19 million** in the test set. Handily the organisers have provided a `train_sample.csv` which contains 100K rows in case you don't want to download the full data
# 
# For this analysis, I'm going to use the first 1M rows of the training and test datasets.
# 
# ## Data overview

# In[20]:


df_train = pd.read_csv('../input/train.csv', nrows=1000000)
df_test = pd.read_csv('../input/test.csv', nrows=1000000)


# In[24]:


print('Training set:')
df_train.head()


# In[26]:


print('Test set:')
df_test.head()


# ### Looking at the columns
# 
# According to the data page, our data contains:
# 
# * `ip`: ip address of click
# * `app`: app id for marketing
# * `device`: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# * `os`: os version id of user mobile phone
# * `channel`: channel id of mobile ad publisher
# * `click_time`: timestamp of click (UTC)
# * `attributed_time`: if user download the app for after clicking an ad, this is the time of the app download
# * `is_attributed`: the target that is to be predicted, indicating the app was downloaded
# 
# **A few things of note:**
# * If you look at the data samples above, you'll notice that all these variables are encoded - meaning we don't know what the actual value corresponds to - each value has instead been assigned an ID which we're given. This has likely been done because data such as IP addresses are sensitive, although it does unfortunately reduce the amount of feature engineering we can do on these.
# * The `attributed_time` variable is only available in the training set - it's not immediately useful for classification but it could be used for some interesting analysis (for example, one could fill in the variable in the test set by building a model to predict it).
# 
# For each of our encoded values, let's look at the number of unique values:

# In[53]:


plt.figure(figsize=(15, 8))
cols = ['ip', 'app', 'device', 'os', 'channel']
uniques = [len(df_train[col].unique()) for col in cols]
sns.set(font_scale=1.2)
ax = sns.barplot(cols, uniques, palette=pal, log=True)
ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')
for p, uniq in zip(ax.patches, uniques):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            uniq,
            ha="center") 
# for col, uniq in zip(cols, uniques):
#     ax.text(col, uniq, uniq, color='black', ha="center")


# ##  Encoded variables statistics
# 
# Although the actual values of these variables aren't helpful for us, it can still be useful to know what their distributions are. Note these statistics are computed on 1M samples, and so will be higher for the full dataset.

# In[79]:


for col, uniq in zip(cols, uniques):
    counts = df_train[col].value_counts()

    sorted_counts = np.sort(counts.values)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line, = ax.plot(sorted_counts, color='red')
    ax.set_yscale('log')
    plt.title("Distribution of value counts for {}".format(col))
    plt.ylabel('log(Occurence count)')
    plt.xlabel('Index')
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.hist(sorted_counts, bins=50)
    ax.set_yscale('log', nonposy='clip')
    plt.title("Histogram of value counts for {}".format(col))
    plt.ylabel('Number of IDs')
    plt.xlabel('Occurences of value for ID')
    plt.show()
    
    max_count = np.max(counts)
    min_count = np.min(counts)
    gt = [10, 100, 1000]
    prop_gt = []
    for value in gt:
        prop_gt.append(round((counts > value).mean()*100, 2))
    print("Variable '{}': | Unique values: {} | Count of most common: {} | Count of least common: {} | count>10: {}% | count>100: {}% | count>1000: {}%".format(col, uniq, max_count, min_count, *prop_gt))
    


# ## What we're trying to predict

# In[63]:


plt.figure(figsize=(8, 8))
sns.set(font_scale=1.2)
mean = (df_train.is_attributed.values == 1).mean()
ax = sns.barplot(['Fraudulent (1)', 'Not Fradulent (0)'], [mean, 1-mean], palette=pal)
ax.set(xlabel='Target Value', ylabel='Probability', title='Target value distribution')
for p, uniq in zip(ax.patches, [mean, 1-mean]):
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height+0.01,
            '{}%'.format(round(uniq * 100, 2)),
            ha="center") 


# Wow, that's a really unbalanced dataset. Only 0.2% of the dataset is made up of fradulent clicks. This means that any models we run on the data will either need to be robust against class imbalance or will require some data resampling.
