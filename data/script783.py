
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Get first 10000 rows and print some info about columns
train = pd.read_csv("../input/train.csv", parse_dates=['srch_ci', 'srch_co'], nrows=10000)
train.info()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
# preferred continent destinations
sns.countplot(x='hotel_continent', data=train)


# In[ ]:


# most of people booking are from continent 3 I guess is one of the rich continent?
sns.countplot(x='posa_continent', data=train)


# In[ ]:


# putting the two above together
sns.countplot(x='hotel_continent', hue='posa_continent', data=train)


# In[ ]:


# how many people by continent are booking from mobile
sns.countplot(x='posa_continent', hue='is_mobile', data = train)


# In[ ]:


# Difference between user and destination country
sns.distplot(train['user_location_country'], label="User country")
sns.distplot(train['hotel_country'], label="Hotel country")
plt.legend()


# In[ ]:


import numpy as np
# get number of booked nights as difference between check in and check out
hotel_nights = train['srch_co'] - train['srch_ci'] 
hotel_nights = (hotel_nights / np.timedelta64(1, 'D')).astype(float) # convert to float to avoid NA problems
train['hotel_nights'] = hotel_nights
plt.figure(figsize=(11, 9))
ax = sns.boxplot(x='hotel_continent', y='hotel_nights', data=train)
lim = ax.set(ylim=(0, 15))


# In[ ]:


plt.figure(figsize=(11, 9))
sns.countplot(x="hotel_nights", data=train)


# In[ ]:


# distribution of the total number of people per cluster
src_total_cnt = train.srch_adults_cnt + train.srch_children_cnt
train['src_total_cnt'] = src_total_cnt
ax = sns.kdeplot(train['hotel_cluster'], train['src_total_cnt'], cmap="Purples_d")
lim = ax.set(ylim=(0.5, 4.5))


# In[ ]:


# plot all columns countplots
import numpy as np
rows = train.columns.size//3 - 1
fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(12,18))
fig.tight_layout()
i = 0
j = 0
for col in train.columns:
    if j >= 3:
        j = 0
        i += 1
    # avoid to plot by date    
    if train[col].dtype == np.int64:
        sns.countplot(x=col, data=train, ax=axes[i][j])
        j += 1

